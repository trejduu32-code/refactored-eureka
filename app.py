import time
import gradio as gr
from openai import OpenAI

def format_time(seconds_float):
    total_seconds = int(round(seconds_float))
    hours = total_seconds // 3600
    remaining_seconds = total_seconds % 3600
    minutes = remaining_seconds // 60
    seconds = remaining_seconds % 60
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

DESCRIPTION = '''
# Duplicate the space for free private inference.
## Qwen3-0.6B Demo
A reasoning model trained using RL (Reinforcement Learning) that demonstrates structured reasoning capabilities.
'''

CSS = """
.spinner {
    animation: spin 1s linear infinite;
    display: inline-block;
    margin-right: 8px;
}
@keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}
.thinking-summary {
    cursor: pointer;
    padding: 8px;
    background: #f5f5f5;
    border-radius: 4px;
    margin: 4px 0;
}
.thought-content {
    padding: 10px;
    background: #f8f9fa;
    border-radius: 4px;
    margin: 5px 0;
}
.thinking-container {
    border-left: 3px solid #facc15;
    padding-left: 10px;
    margin: 8px 0;
    background: #210c29;
}
details:not([open]) .thinking-container {
    border-left-color: #290c15;
}
details {
    border: 1px solid #e0e0e0 !important;
    border-radius: 8px !important;
    padding: 12px !important;
    margin: 8px 0 !important;
    transition: border-color 0.2s;
}
"""

client = OpenAI(base_url="http://localhost:8080/v1", api_key="no-key-required")

def user(message, history):
    return "", history + [[message, None]]

class ParserState:
    __slots__ = ['answer', 'thought', 'in_think', 'start_time', 'last_pos', 'total_think_time']
    def __init__(self):
        self.answer = ""
        self.thought = ""
        self.in_think = False
        self.start_time = 0
        self.last_pos = 0
        self.total_think_time = 0.0

def parse_response(text, state):
    buffer = text[state.last_pos:]
    state.last_pos = len(text)
    
    while buffer:
        if not state.in_think:
            think_start = buffer.find('<think>')
            if think_start != -1:
                state.answer += buffer[:think_start]
                state.in_think = True
                state.start_time = time.perf_counter()
                buffer = buffer[think_start + 7:]
            else:
                state.answer += buffer
                break
        else:
            think_end = buffer.find('</think>')
            if think_end != -1:
                state.thought += buffer[:think_end]
                # Calculate duration and accumulate
                duration = time.perf_counter() - state.start_time
                state.total_think_time += duration
                state.in_think = False
                buffer = buffer[think_end + 8:]
            else:
                state.thought += buffer
                break
    
    elapsed = time.perf_counter() - state.start_time if state.in_think else 0
    return state, elapsed

def format_response(state, elapsed):
    answer_part = state.answer.replace('<think>', '').replace('</think>', '')
    collapsible = []
    collapsed = "<details open>"

    if state.thought or state.in_think:
        if state.in_think:
            # Ongoing think: total time = accumulated + current elapsed
            total_elapsed = state.total_think_time + elapsed
            formatted_time = format_time(total_elapsed)
            status = f"🌀 Thinking for {formatted_time}"
        else:
            # Finished: show total accumulated time
            formatted_time = format_time(state.total_think_time)
            status = f"✅ Thought for {formatted_time}"
            collapsed = "<details>"
        collapsible.append(
            f"{collapsed}<summary>{status}</summary>\n\n<div class='thinking-container'>\n{state.thought}\n</div>\n</details>"
        )

    return collapsible, answer_part

def generate_response(history, temperature, top_p, max_tokens, active_gen):
    messages = [{"role": "user", "content": history[-1][0]}]
    full_response = ""
    state = ParserState()
    last_update = 0
    
    try:
        stream = client.chat.completions.create(
            model="",
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=True
        )
        
        for chunk in stream:
            if not active_gen[0]:
                break
            
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                state, elapsed = parse_response(full_response, state)
                
                collapsible, answer_part = format_response(state, elapsed)
                history[-1][1] = "\n\n".join(collapsible + [answer_part])
                yield history
        
        # Final update to ensure all content is parsed
        state, elapsed = parse_response(full_response, state)
        collapsible, answer_part = format_response(state, elapsed)
        history[-1][1] = "\n\n".join(collapsible + [answer_part])
        yield history
        
    except Exception as e:
        history[-1][1] = f"Error: {str(e)}"
        yield history
    finally:
        active_gen[0] = False

with gr.Blocks(css=CSS) as demo:
    gr.Markdown(DESCRIPTION)
    active_gen = gr.State([False])
    
    chatbot = gr.Chatbot(
        elem_id="chatbot",
        height=500,
        show_label=False,
        render_markdown=True
    )

    with gr.Row():
        msg = gr.Textbox(
            label="Message",
            placeholder="Type your message...",
            container=False,
            scale=4
        )
        submit_btn = gr.Button("Send", variant='primary', scale=1)
    
    with gr.Column(scale=2):
        with gr.Row():
            clear_btn = gr.Button("Clear", variant='secondary')
            stop_btn = gr.Button("Stop", variant='stop')
        
        with gr.Accordion("Parameters", open=False):
            temperature = gr.Slider(minimum=0.1, maximum=1.5, value=0.6, label="Temperature")
            top_p = gr.Slider(minimum=0.1, maximum=1.0, value=0.95, label="Top-p")
            max_tokens = gr.Slider(minimum=2048, maximum=32768, value=4096, step=64, label="Max Tokens")

    gr.Examples(
        examples=[
            ["Can you help me understand gravity like i am 5!"],
            ["Write 10 funny sentences that end in a fruit name!"],
            ["Let’s play word chains! I’ll start: PIZZA. Your turn! Next word must start with… A!"]
        ],
        inputs=msg,
        label="Example Prompts"
    )
    
    submit_event = submit_btn.click(
        user, [msg, chatbot], [msg, chatbot], queue=False
    ).then(
        lambda: [True], outputs=active_gen
    ).then(
        generate_response, [chatbot, temperature, top_p, max_tokens, active_gen], chatbot
    )
    
    msg.submit(
        user, [msg, chatbot], [msg, chatbot], queue=False
    ).then(
        lambda: [True], outputs=active_gen
    ).then(
        generate_response, [chatbot, temperature, top_p, max_tokens, active_gen], chatbot
    )
    
    stop_btn.click(
        lambda: [False], None, active_gen, cancels=[submit_event]
    )
    
    clear_btn.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
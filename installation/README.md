# Ollama

- Download and install ollama from 	[here](https://ollama.ai/download)
- Fetch available LLM model via ollama pull <name-of-model>
    - View a list of available models via the [model library](https://ollama.ai/library)
    
        `ollama pull llama2`

## Usage - as chat GPT
- Open a Terminal
- After pulling the model, run the model locally using below command

    `ollama run llama2`

- The terminal will open a shell through which you can interact with and ask question locally like a chat GPT


## Usage - as Embedding provider

    curl http://localhost:11434/api/embeddings -d '{
        "model": "llama2",
        "prompt": "<Query text to be embedded>"
        }'

## Usage - as LLM

    curl http://localhost:11434/api/chat -d '{
        "model": "llama2",
        "messages": [
            {
                "role": "user",
                "content": "<Your question>"
            }
                    ],
        "stream": false
    }'

Can explore from [Ollama Git Repository](https://github.com/ollama/ollama/blob/main/docs/api.md)





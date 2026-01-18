### Your RLRA Agent to help you!

To start, follow the steps to setup the environment and Ollama. 
Then to setup the base and start asking questions:

```
python run.py
```

Tell the agent to what you are looking for by giving the title or content. 
Then tell the type of other information, you are looking for :
1 - Code 
2 - Method 
3 - Results 
4 - Methodology


### Setup Environment and OLLAMA 


```
    conda env create -f environment.yml
    curl -fsSL https://ollama.com/install.sh | sh
    ollama serve
    ollama pull llama3.1
```


### Current Knowledge Base include the following papers :

1. Foundation Policies with Hilbert Representations
2. Reinforcement Learning from Passive Data via Latent Intentions
3. Offline Reinforcement Learning with Implicit Q-Learning
4. Does Zero-Shot Reinforcement Learning Exist?
5. Reinforcement Learning with Prototypical Representations


### To Do:

1. Compare the results with individual paper retrieval. Get 5 results and compare the answers.
2. Compare results with different prompts.
3. Use Cloud-based VS called only when change in the knowledge base is detected.

### Learning Objectives 

1. When to choose RAG over Semantic Search over VS?
2. Is combined knowledge base also better?
3. How to pick the right index for the vector store?
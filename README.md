
![Alt text](https://github.com/amr-sheriff/arxiv-assistant/blob/main/public/logo_dark.png)
<p align="center">A RAG Assistant Prototype. The live app can be accessed at </p>[arXiv Assistant](https://amrsherif.live/arxiv-assistant).

---
This project is a demonstration of how to prototype a Retrieval-Augmented Generation (RAG) Assistant employing a suite of open-source technologies, frameworks and fine-tuned Large Language Models.

It can be adapted to many other business-specific use cases. 

## Demo 🎥

The arXiv Assistant is a simple demo designed to help researchers and practitioners stay up-to-date with the latest advancements in various fields by retrieving, summarizing, and answering queries about research papers from arXiv.

The assistant can retrieve and select relevant research papers based on user-specified criteria such as submission/revision date, domain/category, and topic. Additionally, it can also answer questions about the papers and highlight key points. 

## Development Process 🛠️

The development process involved leveraging state-of-the-art techniques to implement a lightweight yet efficient LLM system. Key technologies used include:
- [**Instruction Tuning Dataset**](https://huggingface.co/datasets/amrachraf/arXiv-full-text-synthetic-instruct-tune): Generate domain-specific synthetic dataset consisting of instructions and QAs pairs.
- **QLoRA Quantization**: For efficient memory usage.
- **Parameter-Efficient Fine-Tuning (PEFT)**: Utilizing LoRA adapters. PEFT mdel available [here](https://huggingface.co/amrachraf/arxiv-assistant-merged_peft_model).
- **Function Calling and In-Context Learning**: To enhance the assistant's capabilities.
- **Retrieval Augmented Generation (RAG) process**: To improve contextual understanding and relevance, which enhances the retrieval and generation process.
- **vLLM**: As the serving and inference engine.
- **HuggingFace Text Embedding Inference**: Serving the embedding model using HuggingFace to provide high-quality embeddings for document retrieval and processing.

### Key Features ✨

- **Retrieve and Summarize Papers**: Quickly find and get summaries of the latest research papers.
- **Answer Queries**: Get answers to specific questions about research papers.
- **Web-Based UI**: Built with Chainlit for an interactive user experience.
- **Observability Functionality**: Supported by Literal AI, providing insights and monitoring for the assistant's performance and operations.
- **In-Chat Memory**: Allows the assistant to remember previous interactions within a session.
- **Resume Chat Capability**: Enables users to continue previous chat sessions seamlessly.
- **Chain of Thought Visualization**: Supported by Literal AI, providing an intuitive understanding of the assistant's reasoning process.
- **Data persistence & human feedback**: Ensures that the data is retained and allows for continuous improvement through user feedback.

### Integrations & Frameworks 🔌

- [**Weights & Biases**](https://wandb.ai/site): For monitoring and logging the fine-tuning process.
- [**LangChain**](https://www.langchain.com/): For implementing retrieval and processing modules.
- [**HuggingFace TEI**](https://huggingface.co/docs/text-embeddings-inference/en/index): For text embedding inference, serving high-quality embeddings.
- [**vLLM**](https://github.com/vllm-project/vllm): As the serving and inference engine.
- [**Chainlit**](https://github.com/Chainlit/chainlit/tree/main): to build scalable conversational AI or agentic applications.
- [**Literal AI**](https://literalai.com/): LLM evaluation and observability platform

## Quickstart Guide ⚡

To get started with arXiv Assistant, open the terminal and follow these steps:

1. **Clone the Repository**:
   ```bash
   $ git clone https://github.com/your-repo/arxiv-assistant.git
   $ cd arxiv-assistant
   ```

2. **Build docker image**:
   ```bash
   $ docker build -t arxiv-assistant:latest .
   ```

3. **Run countainer**:
   ```bash
   $ docker run -d --env-file .env -p 8080:8080 arxiv_assistant:latest
   ```

Run the app locally and navigate to [localhost:8080/arxiv-assistant](http://localhost:8080/arxiv-assistant) 🥂
 
## License 📜
This project is licensed under the [Apache 2.0](https://github.com/amr-sheriff/arxiv-assistant/blob/main/LICENSE) license.



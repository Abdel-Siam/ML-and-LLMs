# **Large Language Models and Machine Learning**
# Intro
Hey Calvin, hope you are doing well. I thought I would put all of these resources here as you look more into the world of ML and LLMs. It should be a very interesting thing to do now that you are retired.

##### NOTE
The more I have used LLMs like ChatGPT, Claude 3.5 Sonnet, and Gemini, I have come to the realization that LLMs are just function approximators that can focus on words. These massive models have been trained on a lot of human data, including human code. So at the end of the day, when you look at their output, do not be surprised that it is not the best because...

# Table of contents
- [TLDR (Too Long Did not Read)](#tldr-too-long-did-not-read)
- [Background Information](#background-information)
  - [Tokens](#tokens)
  - [Models](#models)
  - [Backends](#backends)
  - [Quantization (model Types)](#quantization-model-types)
- [Starter Pack](#starter-pack)
  - [Basics](#basics)
  - [Basic Local LLM setup](#basic-local-llm-setup)
- [LLM Knowledge Sources](#llm-knowledge-sources)
  - [Reddit Communities](#reddit-communities)
  - [YouTube Channels](#youtube-channels)
  - [Websites](#websites)
- [Github Repos](#github-repos)
  - [WebUIs](#webuis)
  - [LLMs Systems and Frameworks](#llms-systems-and-frameworks)
  - [The one I made](#the-one-i-made)
- [Thank You for reading!](#thank-you-for-reading)

# TLDR (Too Long Did not Read)
Jump to [Basic Local LLM setup](#basic-local-llm-setup)

# Background Information

## Tokens:
A token is the most basic unit that is inputted to an LLM model. It is a word or part of the word in the English language (or the wanted language) that is translated into a numerical value that will be inputted into the model. You can test out a tokenizer here, change the input and different things should show up for different models:
https://www.danieldemmel.me/tokenizer.html

## Models:
There are 2 main groups for AI Models:

1. Closed Source Models
   - Examples: OpenAI ChatGPT and Anthropic Claude
   - To access these models, you would need to get an API key from them, and then you can run these models. They are pretty cheap and offer better performance than almost all of the open source models.
   - They are not that expensive depending on the model used and the amount of tokens you provide in each prompt.

2. "Open" Source Models
   - Examples: the LLama Family of models, the latest is LLama 3, all from Meta (Facebook)
   - Another Example is Mistal, which is a French company that released a couple of models as open source. (Google and Microsoft have also released some open source models)
   - There are many others. These models might not be the best, but they offer pretty good performance for their size and speed. They can also be fine-tuned to perform specific tasks really well, even better than the original model or closed source models.
   - There are so many uses for these open source models, and they are genuinely impressive. But overall, they are limited by your own hardware and what you can run them on.

To download open source models, you can check them out at https://huggingface.co/. It's the best and biggest place for all your ML model needs. There are other places like https://www.kaggle.com/, but they are not as big. The biggest benefit of the open source models is that they can be tailored to your needs and they offer privacy, which the closed source models do not.

## Backends
A backend is the framework for running the LLM Models. These backends have different properties and run differently from one another. Each has some different requirements for the types of models (quantization) that they can run. If your goal is to quickly run and setup LLMs on your own computer, go [here](#basic-local-llm-setup) as these are mainly for if you wanted to code a project for yourself. Here is a quick list of them and what they can do:

- [Transformers](https://huggingface.co/docs/transformers/en/index) from Hugging Face is the easiest one to setup and get started in your Python code.
- [llamacpp](https://github.com/ggerganov/llama.cpp) is the most advanced backend. It's written in C and C++ but there is a [Python wrapper](https://github.com/abetlen/llama-cpp-python) for it. It usually has the newest features for models and has a lot of compatibility.
- [vLLM](https://github.com/vllm-project/vllm) or [LLMDeploy](https://github.com/InternLM/lmdeploy#quantization) I think currently these are the fastest backends to run LLMs. They do require a specific quantization and newer NVIDIA GPUs with a certain [CUDA compatibility](https://developer.nvidia.com/cuda-gpus) - Click on Enabled GeForce as those are the consumer desktop graphics cards.

## Quantization (model Types)
Alright Calvin, let's talk about quantization - it's a pretty important concept when you're dealing with LLMs. Basically, quantization is a technique used to reduce the size of models without losing too much performance. Here's the lowdown:

- **Full Precision (FP32)**: This is the original, uncompressed model. It's the biggest and most accurate, but also the most resource-hungry.
- **Half Precision (FP16)**: This cuts the model size roughly in half. It's a good balance between size and accuracy for many applications.
- **INT8**: This quantizes the model to 8-bit integers. It's significantly smaller than FP16 and can run faster, but there might be a slight drop in quality.
- **INT4**: Taking it a step further, this quantizes to 4-bit integers. It's even smaller and faster, but you'll likely see a more noticeable drop in performance.
- **GGUF**: This is a file format for quantized models, particularly popular with llama.cpp. It supports various quantization levels and is known for its efficiency and compatibility across different systems.
- **GPTQ**: This is a fancy quantization method that can get models down to 4-bit or even 3-bit sizes while maintaining surprisingly good performance.
- **AWQ**: Another advanced quantization technique that can achieve similar compression to GPTQ, sometimes with better results.
- **ExLlamaV2**: This is both a quantization format and a custom GPU inference engine (similar to llamacpp). It's known for its speed and efficiency, especially on consumer-grade GPUs. It can run models at very low bit depths (like 2-3 bits per parameter) while maintaining good performance.

The cool thing about quantization is that it allows you to run larger models on less powerful hardware. For example, you might be able to run a 7B parameter model quantized to INT4 on a GPU that couldn't handle the full FP32 version.

When you're looking at models on Hugging Face or elsewhere, you'll often see these quantization types mentioned. It's worth experimenting with different quantizations to find the sweet spot between model size, speed, and performance for your specific use case.

Remember, the effectiveness of quantization can vary depending on the model and the task. Sometimes a heavily quantized model will perform almost as well as the full version, while other times you might notice a significant difference. It's all about finding the right balance for what you're trying to do! But overall, any quantization will always have a degradation in the model's performance; it's just how much of a degradation is acceptable to you.

# Starter Pack
This starter pack contains links to things that you might need to get you started.

## Basics
- https://github.com/microsoft/generative-ai-for-beginners an absolute treasure of information about everything that you might need for LLM basics
- https://www.kaggle.com/ A massive ML community. There are datasets and Models that people created and even challenges and hackathons. There are also GPU Notebooks
- https://www.promptingguide.ai/ Prompting guide on how to write prompts for different things and even some examples

## Basic Local LLM setup
For this, you would need a graphics card with some amount of video memory (VRAM) to be able to run it at reasonable speeds. The amount of video memory depends on the model's size.

**YOU CAN RUN THIS WITH JUST THE CPU**. You will need a good amount of RAM and it will be a lot slower, but it is possible and people use it a lot of the time to run bigger models. You can even use both the CPU and GPU to run a portion of the model in the GPU and the remaining on the CPU. This would allow for some speed up from just using the CPU and the ability to load bigger models than if you just used the GPU.

- [Ollama](https://ollama.com/) The easiest to install backend 
- A compatible front end would be [OpenWebUI](https://docs.openwebui.com/) This is an amazing project and the team behind it is great. It's open source and has the ability to work with open and closed models (there are places to input API keys). You can have this automatically install Ollama for you.

I have listed some other Web UIs in the [Github Repos](#github-repos) section, but these 2 are the easiest thing to do to get started and test out things. OpenWebUI even has the ability to upload documents and have the LLM get asked from them. Ollama also provides you with an API endpoint from which you can run the LLMs locally and easily.

# LLM Knowledge Sources

### Reddit Communities
These are the main Reddit channels that I visit. News shows up here very quickly, especially when something interesting happens, and people usually ask questions which are helpful.
- https://www.reddit.com/r/LocalLLaMA/
- https://www.reddit.com/r/MachineLearning/

### YouTube Channels
These are in no particular order. They are channels I like to check out and watch their content for either staying up to date with the latest LLM information or to learn about some Machine Learning concepts.

- [AI Explained](https://www.youtube.com/@aiexplained-official): Great channel that covers topics of LLMs from an objective point of view that I really found refreshing
- [3Blue1Brown](https://www.youtube.com/@3blue1brown): Amazing channel for educational content. He also has some videos about LLMs and it's very easy to understand how they work because of visualization. He also has other topics about Neural Networks and machine learning.
- [TwoMinutePapers](https://www.youtube.com/@TwoMinutePapers): While his name is 2 min papers, he actually takes longer to cover them, but he looks into new ML papers
- [ByCloud](https://www.youtube.com/@bycloudAI): Pretty good channel with easy to understand videos about new LLM things and standard things. He does talk very quickly though.

#### Specific Videos:
- https://www.youtube.com/watch?v=kCc8FmEb1nY This video is a very detailed explanation of creating GPT-2 from scratch. He builds everything from the bottom up. It's almost 2 hours.

### Websites
Some websites that have some good content.
- [Daily Papers](https://huggingface.co/papers): New ML and LLM papers. They are free to check out
- [Gradio](https://www.gradio.app/) A Python library that helps you make a UI for ML and LLMs very easily and quickly. It's what I have been using a lot on my projects 
- [RAG Explanation](https://www.pinecone.io/learn/series/rag/rerankers/) explanation of RAG and RERANK. These are 2 things that have been developed to help overcome some of the limitations of LLMs and this article does a pretty good job of explaining it.

# Github Repos
Here are some interesting Github repos that I have checked out. There are some good ones.

## WebUIs
- [Text-generation-Webui](https://github.com/oobabooga/text-generation-webui) Another webui for LLMs but this one was one of the first ones to be created for LLMs and is still going strong. It's not as polished as OpenWebUI but this one is for more advanced users as it allows for more tinkering with the system prompt, different backends and has the most amount of support for model types. On their readme you can see them even saying "Multiple backends for text generation in a single UI and API, including [Transformers](https://github.com/huggingface/transformers), [llama.cpp](https://github.com/ggerganov/llama.cpp) (through [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)), [ExLlamaV2](https://github.com/turboderp/exllamav2), [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ), and [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM). [AutoAWQ](https://github.com/casper-hansen/AutoAWQ), [HQQ](https://github.com/mobiusml/hqq), and [AQLM](https://github.com/Vahe1994/AQLM) are also supported through the Transformers loader."
- [SillyTavern](https://github.com/SillyTavern/SillyTavern) A webui for LLMs while meant for roleplay has a really good RAG system

## LLMs Systems and Frameworks
- [RAG from scratch](https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_1_to_4.ipynb) How to build a Retrieval Augmented system from scratch, where it grabs relevant text from a large number of texts from documents. 
- [dsRAG](https://github.com/D-Star-AI/dsRAG) A RAG retrieval engine for unstructured data. Haven't looked into it much as I haven't had enough time but it seems very interesting

## The one I made
Now this is the UI that I had made that used OpenAI API endpoints. My main goal was to not have to spend $20 USD a month on a subscription and only use what I want. Also, their APIs provided newer models than what was available on their website, so they were much better. I also had added features like PDF parsing and image uploading before they had allowed them on the website. I am not sure if this still even works. The libraries that I had built this on have changed drastically since then and I had built my own component to the **Gradio** library for allowing better text input. Which after a while they actually made their own so I had to deprecate mine as it was basically redundant now. You can see it [here](https://pypi.org/project/gradio-bettertextbox/) 
- https://github.com/Abdel-Siam/As-You-Go-GPT 

The code isn't the best and I could have definitely improved it a lot more. The comments are bad as they explain what the code does, not why it is. I have learned a lot of things during my time at LH and if I had to revisit this project I would have done it very differently, but when I was making it things were different and the field was moving very quickly too.

# Thank You for reading!

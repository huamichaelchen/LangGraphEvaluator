import json
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def create_research_evaluation_dataset() -> List[Dict[str, Any]]:
    """
    Create a comprehensive synthetic dataset for evaluating the research assistant.
    
    This dataset covers various research scenarios including:
    - Literature review requests
    - Specific paper searches
    - Multi-turn research conversations
    - Methodology questions
    - Citation requests
    """
    
    dataset = [
        # Literature Review Scenarios
        {
            "inputs": {
                "question": "Can you help me conduct a literature review on recent advances in large language model evaluation methods?"
            },
            "outputs": {
                "answer": "I'll help you conduct a comprehensive literature review on recent advances in large language model evaluation methods. This is a rapidly evolving field with several key areas of development.\n\n**Key Areas of Recent Advancement:**\n\n1. **Automated Evaluation Metrics**\n   - Recent work has focused on developing more sophisticated metrics beyond BLEU and ROUGE\n   - Papers like 'BERTScore' and 'MoverScore' have introduced semantic similarity measures\n   - The development of task-specific evaluation frameworks\n\n2. **Human Evaluation Frameworks**\n   - Standardized protocols for human evaluation of LLM outputs\n   - Crowdsourcing methodologies for large-scale evaluation\n   - Inter-annotator agreement studies\n\n3. **Benchmark Datasets**\n   - Introduction of comprehensive benchmarks like GLUE, SuperGLUE, and BIG-bench\n   - Domain-specific evaluation suites\n   - Multilingual evaluation frameworks\n\n**Recommended Starting Points:**\n- Search for recent papers on \"language model evaluation\" on arXiv\n- Look for benchmark papers and evaluation framework publications\n- Consider papers from major AI conferences (NeurIPS, ICML, ACL, EMNLP)\n\nWould you like me to search for specific recent papers in any of these areas?"
            },
            "metadata": {
                "scenario": "literature_review",
                "complexity": "high",
                "expected_tools": ["arxiv_search", "web_search"]
            }
        },
        
        # Specific Paper Search
        {
            "inputs": {
                "question": "Find recent papers about graph neural networks for molecular property prediction published in the last year."
            },
            "outputs": {
                "answer": "I'll search for recent papers on graph neural networks for molecular property prediction. This is an active area of research combining graph neural networks with computational chemistry.\n\n**Key Research Areas:**\n- Graph neural network architectures for molecular representation\n- Property prediction tasks (solubility, toxicity, drug-likeness)\n- Benchmarking datasets and evaluation protocols\n- Integration with quantum chemical calculations\n\n**Expected Recent Developments:**\n- New GNN architectures specifically designed for molecular graphs\n- Improved training methodologies and data augmentation techniques\n- Applications to drug discovery and materials science\n- Comparative studies of different molecular representation methods\n\nLet me search arXiv for recent publications in this area."
            },
            "metadata": {
                "scenario": "paper_search",
                "complexity": "medium",
                "expected_tools": ["arxiv_search"]
            }
        },
        
        # Multi-turn Research Conversation
        {
            "inputs": {
                "question": "What are the main challenges in training large language models?"
            },
            "outputs": {
                "answer": "Training large language models faces several significant challenges across multiple dimensions:\n\n**1. Computational Challenges**\n- Massive computational requirements (thousands of GPUs for months)\n- Memory constraints and distributed training complexities\n- Energy consumption and environmental impact\n- Hardware optimization and efficiency considerations\n\n**2. Data Challenges**\n- Need for high-quality, diverse training data at scale\n- Data preprocessing and cleaning at petabyte scale\n- Handling bias and toxicity in training data\n- Copyright and licensing issues with training data\n\n**3. Technical Challenges**\n- Gradient instability and training dynamics\n- Optimization difficulties in high-dimensional spaces\n- Catastrophic forgetting and knowledge retention\n- Scaling laws and emergent behaviors\n\n**4. Evaluation and Safety**\n- Difficulty in comprehensive evaluation\n- Alignment with human values and intentions\n- Safety concerns and potential misuse\n- Robustness and reliability issues\n\n**5. Economic and Resource Challenges**\n- High costs of training and inference\n- Resource accessibility and democratization\n- Sustainability considerations\n\nWould you like me to dive deeper into any of these specific challenge areas, or search for recent research addressing these issues?"
            },
            "metadata": {
                "scenario": "multi_turn_research",
                "complexity": "high",
                "expected_tools": ["web_search", "arxiv_search"]
            }
        },
        
        # Methodology Questions
        {
            "inputs": {
                "question": "What are the best practices for conducting reproducible research in machine learning?"
            },
            "outputs": {
                "answer": "Reproducible research in machine learning requires systematic approaches across the entire research pipeline. Here are the key best practices:\n\n**1. Code and Environment Management**\n- Use version control systems (Git) for all code\n- Provide complete environment specifications (requirements.txt, conda environments)\n- Containerization using Docker for consistent environments\n- Clear documentation of dependencies and versions\n\n**2. Data Management**\n- Version control for datasets with checksums\n- Clear data provenance and preprocessing steps\n- Separate raw data from processed data\n- Document data collection and filtering procedures\n\n**3. Experiment Tracking**\n- Log all hyperparameters, model architectures, and training configurations\n- Use experiment tracking tools (MLflow, Weights & Biases, TensorBoard)\n- Record random seeds and ensure deterministic behavior\n- Track computational resources and training times\n\n**4. Model and Results Management**\n- Save model checkpoints and final trained models\n- Include model evaluation code and metrics\n- Provide statistical significance testing\n- Document hardware specifications used\n\n**5. Documentation and Reporting**\n- Comprehensive README files with setup instructions\n- Clear code comments and documentation\n- Detailed methodology in papers\n- Include failure cases and limitations\n\n**6. Open Science Practices**\n- Share code repositories publicly\n- Provide pre-trained models when possible\n- Use standard evaluation protocols and benchmarks\n- Participate in reproducibility challenges\n\nWould you like me to find specific papers or resources about reproducibility in ML research?"
            },
            "metadata": {
                "scenario": "methodology",
                "complexity": "medium",
                "expected_tools": ["web_search"]
            }
        },
        
        # Citation and Reference Requests
        {
            "inputs": {
                "question": "I need proper citations for the original transformer paper and key follow-up works. Can you help me format these correctly?"
            },
            "outputs": {
                "answer": "I'll help you find and format proper citations for the original transformer paper and key follow-up works. The transformer architecture has been foundational to modern NLP.\n\n**Original Transformer Paper:**\n\nVaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008).\n\n**Key Follow-up Works:**\n\n1. **BERT** - Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv preprint arXiv:1810.04805.\n\n2. **GPT** - Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). Improving language understanding by generative pre-training.\n\n3. **T5** - Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., ... & Liu, P. J. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140), 1-67.\n\n4. **Vision Transformer** - Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2020). An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.\n\nWould you like me to search for more recent transformer variants or format these citations in a specific citation style (APA, MLA, Chicago, etc.)?"
            },
            "metadata": {
                "scenario": "citation_formatting",
                "complexity": "medium",
                "expected_tools": ["web_search", "extract_citations"]
            }
        },
        
        # Research Gap Identification
        {
            "inputs": {
                "question": "What are the current research gaps in adversarial robustness for deep learning models?"
            },
            "outputs": {
                "answer": "Adversarial robustness remains a critical challenge in deep learning with several significant research gaps:\n\n**1. Theoretical Understanding Gaps**\n- Limited theoretical frameworks for understanding adversarial vulnerability\n- Lack of provable guarantees for adversarial robustness\n- Insufficient understanding of the robustness-accuracy tradeoff\n- Need for better theoretical analysis of defense mechanisms\n\n**2. Evaluation and Benchmarking Gaps**\n- Inconsistent evaluation protocols across different domains\n- Limited standardized benchmarks for adversarial robustness\n- Difficulty in evaluating robustness against unknown attacks\n- Need for more comprehensive threat models\n\n**3. Real-world Applicability Gaps**\n- Most research focuses on image classification, limited work on other domains\n- Gap between adversarial examples in research and real-world attacks\n- Insufficient consideration of practical deployment constraints\n- Limited work on adversarial robustness in production systems\n\n**4. Scalability and Efficiency Gaps**\n- Computational overhead of robust training methods\n- Scalability issues for large models and datasets\n- Need for efficient adversarial training algorithms\n- Balance between robustness and inference speed\n\n**5. Multi-modal and Cross-domain Gaps**\n- Limited research on adversarial robustness across different modalities\n- Insufficient work on transfer of robustness across domains\n- Need for unified approaches to multi-modal adversarial robustness\n\n**6. Adaptive and Evolving Threats**\n- Defense mechanisms often fail against adaptive attacks\n- Need for defenses that can evolve with new attack methods\n- Insufficient research on long-term robustness\n\nWould you like me to search for recent papers addressing any of these specific gaps?"
            },
            "metadata": {
                "scenario": "research_gaps",
                "complexity": "high",
                "expected_tools": ["arxiv_search", "web_search"]
            }
        },
        
        # Comparative Analysis Request
        {
            "inputs": {
                "question": "Can you compare different approaches to few-shot learning in natural language processing?"
            },
            "outputs": {
                "answer": "I'll provide a comprehensive comparison of different few-shot learning approaches in NLP. This is an active research area with several distinct methodologies:\n\n**1. In-Context Learning (ICL)**\n- **Approach**: Provide examples as part of the input prompt\n- **Strengths**: No parameter updates required, flexible task adaptation\n- **Weaknesses**: Limited by context length, inconsistent performance\n- **Representative Work**: GPT-3 and subsequent large language models\n\n**2. Meta-Learning Approaches**\n- **Approach**: Learn to learn from few examples across tasks\n- **Strengths**: Principled learning framework, good generalization\n- **Weaknesses**: Computationally expensive, requires meta-training data\n- **Representative Work**: MAML, Prototypical Networks adapted for NLP\n\n**3. Prompt-based Learning**\n- **Approach**: Convert tasks into cloze-style prompts\n- **Strengths**: Leverages pre-trained knowledge effectively\n- **Weaknesses**: Sensitive to prompt design, limited task coverage\n- **Representative Work**: PET, LM-BFF, P-tuning\n\n**4. Parameter-Efficient Fine-tuning**\n- **Approach**: Update only a small subset of parameters\n- **Strengths**: Computationally efficient, maintains pre-trained knowledge\n- **Weaknesses**: Task-specific adaptations needed\n- **Representative Work**: LoRA, AdaLoRA, BitFit\n\n**5. Retrieval-Augmented Approaches**\n- **Approach**: Retrieve relevant examples from large databases\n- **Strengths**: Scales with data availability, interpretable\n- **Weaknesses**: Requires good retrieval systems, computational overhead\n- **Representative Work**: kNN-LM, RETRO, RAG\n\n**Performance Comparison:**\n- In-context learning excels with large models but is inconsistent\n- Meta-learning shows strong theoretical foundation but limited scalability\n- Prompt-based methods are highly effective but require careful design\n- Parameter-efficient methods balance performance and efficiency well\n\nWould you like me to search for recent comparative studies or dive deeper into any specific approach?"
            },
            "metadata": {
                "scenario": "comparative_analysis",
                "complexity": "high",
                "expected_tools": ["arxiv_search", "web_search"]
            }
        },
        
        # Technical Implementation Question
        {
            "inputs": {
                "question": "What are the key considerations for implementing distributed training of large neural networks?"
            },
            "outputs": {
                "answer": "Implementing distributed training for large neural networks involves several critical considerations across multiple dimensions:\n\n**1. Parallelization Strategies**\n- **Data Parallelism**: Distribute batches across multiple devices\n- **Model Parallelism**: Split model layers across devices\n- **Pipeline Parallelism**: Divide model into stages for pipeline execution\n- **Hybrid Approaches**: Combine multiple parallelization strategies\n\n**2. Communication and Synchronization**\n- **Gradient Synchronization**: All-reduce, parameter servers, or decentralized approaches\n- **Communication Backends**: NCCL, Gloo, MPI for efficient data transfer\n- **Asynchronous vs. Synchronous**: Trade-offs between speed and consistency\n- **Gradient Compression**: Techniques to reduce communication overhead\n\n**3. Memory Management**\n- **Memory Optimization**: Gradient checkpointing, memory-efficient optimizers\n- **Sharding Strategies**: Distribute model parameters and optimizer states\n- **Dynamic Memory Allocation**: Efficient memory usage across devices\n- **Memory Profiling**: Tools for monitoring and optimizing memory usage\n\n**4. Fault Tolerance and Reliability**\n- **Checkpointing**: Regular saving of training state for recovery\n- **Elastic Training**: Handling dynamic resource allocation\n- **Failure Recovery**: Mechanisms for handling node failures\n- **Monitoring and Logging**: Comprehensive tracking of training progress\n\n**5. Performance Optimization**\n- **Load Balancing**: Ensuring even workload distribution\n- **Bandwidth Optimization**: Minimizing communication bottlenecks\n- **Mixed Precision Training**: Using FP16/BF16 for speed and memory savings\n- **Kernel Optimization**: Custom CUDA kernels for specific operations\n\n**6. Scaling Considerations**\n- **Scaling Laws**: Understanding how performance scales with resources\n- **Batch Size Scaling**: Adjusting learning rates and batch sizes\n- **Communication-Computation Overlap**: Hiding communication latency\n- **Resource Utilization**: Maximizing GPU/TPU utilization\n\n**7. Framework and Infrastructure**\n- **Distributed Frameworks**: PyTorch DDP, Horovod, DeepSpeed, FairScale\n- **Cloud Platforms**: AWS, Google Cloud, Azure distributed training services\n- **Container Orchestration**: Kubernetes for managing distributed jobs\n- **Network Configuration**: High-speed interconnects (InfiniBand, Ethernet)\n\nWould you like me to search for specific papers on any of these aspects or look for recent advances in distributed training frameworks?"
            },
            "metadata": {
                "scenario": "technical_implementation",
                "complexity": "high",
                "expected_tools": ["web_search", "arxiv_search"]
            }
        }
    ]
    
    return dataset

def save_dataset_to_langsmith(dataset: List[Dict[str, Any]], dataset_name: str = "research_assistant_evaluation"):
    """
    Save the evaluation dataset to LangSmith for use in experiments.
    
    Args:
        dataset: List of evaluation examples
        dataset_name: Name for the dataset in LangSmith
    """
    from langsmith import Client
    
    try:
        client = Client()
        
        # Create dataset
        ls_dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="Comprehensive evaluation dataset for research assistant agent covering literature review, paper search, methodology questions, and multi-turn research conversations"
        )
        
        # Convert to LangSmith format
        examples = []
        for item in dataset:
            examples.append({
                "inputs": item["inputs"],
                "outputs": item["outputs"]
            })
        
        # Add examples to dataset
        client.create_examples(
            dataset_id=ls_dataset.id,
            examples=examples
        )
        
        print(f"Successfully created dataset '{dataset_name}' with {len(examples)} examples")
        return ls_dataset
        
    except Exception as e:
        print(f"Error creating dataset: {str(e)}")
        return None

def get_evaluation_scenarios():
    """
    Get different evaluation scenarios for testing various aspects of the research agent.
    """
    scenarios = {
        "literature_review": {
            "description": "Test agent's ability to conduct comprehensive literature reviews",
            "focus_areas": ["search_strategy", "synthesis", "coverage", "relevance"]
        },
        "paper_search": {
            "description": "Test agent's ability to find specific papers and research",
            "focus_areas": ["search_accuracy", "relevance", "recency", "quality"]
        },
        "multi_turn_research": {
            "description": "Test agent's conversational research capabilities",
            "focus_areas": ["context_maintenance", "follow_up", "depth", "coherence"]
        },
        "methodology": {
            "description": "Test agent's knowledge of research methodologies",
            "focus_areas": ["accuracy", "completeness", "best_practices", "clarity"]
        },
        "citation_formatting": {
            "description": "Test agent's ability to handle citations and references",
            "focus_areas": ["format_accuracy", "completeness", "style_consistency"]
        },
        "research_gaps": {
            "description": "Test agent's ability to identify research gaps and opportunities",
            "focus_areas": ["insight_quality", "comprehensiveness", "novelty", "feasibility"]
        },
        "comparative_analysis": {
            "description": "Test agent's ability to compare different approaches or methods",
            "focus_areas": ["fairness", "comprehensiveness", "accuracy", "insight"]
        },
        "technical_implementation": {
            "description": "Test agent's technical knowledge for implementation guidance",
            "focus_areas": ["technical_accuracy", "practicality", "completeness", "clarity"]
        }
    }
    
    return scenarios

if __name__ == "__main__":
    # Create and save the dataset
    dataset = create_research_evaluation_dataset()
    print(f"Created evaluation dataset with {len(dataset)} examples")
    
    # Print dataset summary
    scenarios = {}
    for item in dataset:
        scenario = item["metadata"]["scenario"]
        scenarios[scenario] = scenarios.get(scenario, 0) + 1
    
    print("\nDataset composition:")
    for scenario, count in scenarios.items():
        print(f"  {scenario}: {count} examples")
    
    # Optionally save to LangSmith (requires API key)
    # save_dataset_to_langsmith(dataset)

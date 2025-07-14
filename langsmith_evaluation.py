import os
import json
import re
import sys
from typing import Dict, Any, List
from langsmith import Client, evaluate
from langsmith.evaluation import EvaluationResult
from research_agent import ResearchAgent
from dotenv import load_dotenv
from io import StringIO

# Load environment variables from .env file
load_dotenv()
from evaluators import (
    research_accuracy_evaluator,
    source_credibility_evaluator,
    conversation_helpfulness_evaluator,
    research_depth_evaluator,
    citation_quality_evaluator,
    research_collaboration_summary_evaluator
)
from evaluation_dataset import create_research_evaluation_dataset, save_dataset_to_langsmith
import asyncio
from datetime import datetime

class ResearchAgentEvaluator:
    def __init__(self):
        """Initialize the evaluator with LangSmith client and research agent"""
        self.client = Client()
        self.research_agent = ResearchAgent()
        
        # Evaluator configuration
        self.evaluators = [
            research_accuracy_evaluator,
            source_credibility_evaluator,
            conversation_helpfulness_evaluator,
            research_depth_evaluator,
            citation_quality_evaluator
        ]
        
        self.summary_evaluators = [
            research_collaboration_summary_evaluator
        ]
    
    def prepare_target_function(self):
        """Prepare the target function for evaluation"""
        def target(inputs: Dict[str, Any]) -> Dict[str, Any]:
            """Target function that wraps the research agent"""
            try:
                # Create a unique thread ID for this evaluation
                thread_id = f"eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{hash(str(inputs))}"
                config = {"configurable": {"thread_id": thread_id}}
                
                # Invoke the research agent
                result = self.research_agent.invoke(
                    {"messages": [{"role": "user", "content": inputs["question"]}]},
                    config=config
                )
                
                # Extract the response
                if result and "messages" in result:
                    assistant_message = result["messages"][-1]
                    response_content = assistant_message.content
                    
                    # Extract sources if available
                    sources = []
                    if hasattr(assistant_message, 'additional_kwargs') and 'sources' in assistant_message.additional_kwargs:
                        sources = assistant_message.additional_kwargs['sources']
                    
                    return {
                        "messages": result["messages"],
                        "answer": response_content,
                        "sources": sources
                    }
                else:
                    return {
                        "messages": [],
                        "answer": "No response generated",
                        "sources": []
                    }
                    
            except Exception as e:
                return {
                    "messages": [],
                    "answer": f"Error: {str(e)}",
                    "sources": [],
                    "error": str(e)
                }
        
        return target
    
    def create_or_get_dataset(self, dataset_name: str = "research_assistant_evaluation"):
        """Create or get the evaluation dataset in LangSmith"""
        try:
            # Try to get existing dataset
            existing_datasets = list(self.client.list_datasets(dataset_name=dataset_name))
            if existing_datasets:
                dataset = existing_datasets[0]
                print(f"Using existing dataset: {dataset_name}")
                return dataset
            
            # Create new dataset
            print(f"Creating new dataset: {dataset_name}")
            dataset_examples = create_research_evaluation_dataset()
            
            # Create dataset in LangSmith
            dataset = self.client.create_dataset(
                dataset_name=dataset_name,
                description="Comprehensive evaluation dataset for research assistant agent"
            )
            
            # Add examples
            examples = []
            for item in dataset_examples:
                examples.append({
                    "inputs": item["inputs"],
                    "outputs": item["outputs"]
                })
            
            self.client.create_examples(
                dataset_id=dataset.id,
                examples=examples
            )
            
            print(f"Created dataset with {len(examples)} examples")
            return dataset
            
        except Exception as e:
            print(f"Error creating/getting dataset: {str(e)}")
            return None
    
    def run_sdk_evaluation(self, dataset_name: str = "research_assistant_evaluation"):
        """Run evaluation using LangSmith SDK"""
        print("=" * 60)
        print("RUNNING LANGSMITH SDK EVALUATION")
        print("=" * 60)
        
        try:
            # Get or create dataset
            dataset = self.create_or_get_dataset(dataset_name)
            if not dataset:
                print("Failed to create/get dataset")
                return None
            
            # Prepare target function
            target = self.prepare_target_function()
            
            # Run evaluation
            experiment_name = f"Research_Agent_Evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            print(f"Starting evaluation experiment: {experiment_name}")
            print(f"Dataset: {dataset_name}")
            print(f"Evaluators: {len(self.evaluators)}")
            
            # Capture stdout to get the URL that evaluate() prints
            original_stdout = sys.stdout
            captured_output = StringIO()
            sys.stdout = captured_output
            
            try:
                results = evaluate(
                    target,
                    data=dataset,
                    evaluators=self.evaluators,
                    summary_evaluators=self.summary_evaluators,
                    experiment_prefix=experiment_name,
                    metadata={
                        "version": "1.0",
                        "agent_type": "research_assistant",
                        "evaluation_date": datetime.now().isoformat(),
                        "evaluators": [ev.__name__ for ev in self.evaluators]
                    },
                    max_concurrency=2  # Limit concurrency to avoid rate limits
                )
            finally:
                # Restore stdout
                sys.stdout = original_stdout
                
            # Get the captured output and print it
            console_output = captured_output.getvalue()
            print(console_output)
            
            # Extract the URL from the console output using regex
            url_match = re.search(r'View the evaluation results.*?at:\s*(https://smith\.langchain\.com/[^\s]+)', console_output)
            langsmith_url_from_console = url_match.group(1) if url_match else None
            
            if langsmith_url_from_console:
                print(f"DEBUG: Captured LangSmith URL from console: {langsmith_url_from_console}")
            else:
                print("DEBUG: Could not extract URL from console output")
                print(f"DEBUG: Console output was: {console_output[:500]}...")
            
            print(f"Evaluation completed successfully!")
            print(f"Experiment name: {experiment_name}")
            
            # Wait for results to be fully processed
            results.wait()
            
            # Get the actual experiment name from results (this includes the UUID suffix)
            actual_experiment_name = results.experiment_name if hasattr(results, 'experiment_name') else experiment_name
            print(f"Actual experiment name: {actual_experiment_name}")
            
            # Use the captured URL from console output
            langsmith_url = langsmith_url_from_console
            print(f"DEBUG: Using captured LangSmith URL: {langsmith_url}")
            
            # Debug: Print what we actually got from evaluation
            print(f"\nDEBUG: Results object type: {type(results)}")
            print(f"DEBUG: Results object attributes: {dir(results)}")
            
            # Debug the _manager object to understand URL construction
            if hasattr(results, '_manager'):
                print(f"DEBUG: Manager object: {results._manager}")
                print(f"DEBUG: Manager attributes: {dir(results._manager)}")
                if hasattr(results._manager, 'dataset_id'):
                    print(f"DEBUG: Dataset ID: {results._manager.dataset_id}")
            
            # Try to get aggregate scores from the results
            aggregate_scores = {}
            if hasattr(results, 'aggregate_scores'):
                print("\nAggregate Scores:")
                scores = results.aggregate_scores
                print(f"DEBUG: Aggregate scores type: {type(scores)}")
                print(f"DEBUG: Aggregate scores content: {scores}")
                
                # Try to iterate through scores
                try:
                    for metric, score in scores.items():
                        print(f"  {metric}: {score:.3f}")
                        aggregate_scores[metric] = score
                except Exception as e:
                    print(f"DEBUG: Error iterating scores: {str(e)}")
                    print(f"DEBUG: Raw scores: {scores}")
            else:
                print("DEBUG: No aggregate_scores attribute found")
                # Try to extract scores from individual results
                try:
                    results_list = list(results)
                    if results_list:
                        print(f"DEBUG: Found {len(results_list)} individual results")
                        # Extract scores from feedback
                        for result in results_list:
                            if hasattr(result, 'feedback') and result.feedback:
                                print(f"DEBUG: Found feedback: {result.feedback}")
                except Exception as e:
                    print(f"DEBUG: Error extracting individual results: {str(e)}")
            
            # Extract additional evaluation data
            evaluation_data = {
                'experiment_name': actual_experiment_name,
                'langsmith_url': langsmith_url,
                'results_object': results,
                'console_output': {
                    'dataset_name': dataset_name,
                    'evaluators_count': len(self.evaluators),
                    'completion_status': 'success'
                }
            }
            
            # Add aggregate scores if available
            evaluation_data['aggregate_scores'] = aggregate_scores
            if aggregate_scores:
                print(f"\nStored aggregate scores: {evaluation_data['aggregate_scores']}")
            else:
                print("DEBUG: No aggregate scores to store")
            
            # Try to get additional project data using the experiment name
            try:
                project_data = self.client.read_project(
                    project_name=experiment_name,
                    include_stats=True
                )
                evaluation_data['langsmith_project_data'] = project_data
                print(f"Successfully retrieved LangSmith project data for: {experiment_name}")
            except Exception as e:
                print(f"Could not retrieve LangSmith project data: {str(e)}")
                evaluation_data['langsmith_project_data'] = None
            
            # Debug: Print the final evaluation data structure
            print(f"\nDEBUG: Final evaluation_data structure:")
            print(f"  experiment_name: {evaluation_data.get('experiment_name')}")
            print(f"  langsmith_url: {evaluation_data.get('langsmith_url')}")
            print(f"  aggregate_scores: {evaluation_data.get('aggregate_scores')}")
            print(f"  console_output: {evaluation_data.get('console_output')}")
            
            return evaluation_data
            
        except Exception as e:
            print(f"Error running SDK evaluation: {str(e)}")
            return None
    
    def analyze_results(self, results):
        """Analyze evaluation results and provide insights"""
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS ANALYSIS")
        print("=" * 60)
        
        if not results:
            print("No results to analyze")
            return
        
        try:
            # Aggregate score analysis
            if hasattr(results, 'aggregate_scores'):
                print("\nPerformance Summary:")
                scores = results.aggregate_scores
                
                # Research quality metrics
                research_metrics = {
                    "research_accuracy": scores.get("research_accuracy", 0),
                    "research_depth": scores.get("research_depth", 0),
                    "source_credibility": scores.get("source_credibility", 0)
                }
                
                # Communication metrics
                communication_metrics = {
                    "conversation_helpfulness": scores.get("conversation_helpfulness", 0),
                    "citation_quality": scores.get("citation_quality", 0)
                }
                
                # Calculate overall scores
                research_score = sum(research_metrics.values()) / len(research_metrics)
                communication_score = sum(communication_metrics.values()) / len(communication_metrics)
                overall_score = (research_score + communication_score) / 2
                
                print(f"  Overall Research Quality: {research_score:.3f}")
                print(f"  Overall Communication Quality: {communication_score:.3f}")
                print(f"  Overall Agent Performance: {overall_score:.3f}")
                
                print("\nDetailed Metrics:")
                for metric, score in scores.items():
                    print(f"  {metric}: {score:.3f}")
                
                # Performance interpretation
                print("\nPerformance Interpretation:")
                if overall_score >= 0.8:
                    print("  🟢 Excellent: Agent performs exceptionally well")
                elif overall_score >= 0.6:
                    print("  🟡 Good: Agent performs well with room for improvement")
                elif overall_score >= 0.4:
                    print("  🟠 Fair: Agent shows basic functionality but needs improvement")
                else:
                    print("  🔴 Poor: Agent requires significant improvements")
                
                # Specific recommendations
                print("\nRecommendations:")
                if research_metrics["research_accuracy"] < 0.6:
                    print("  - Improve factual accuracy through better source verification")
                if research_metrics["source_credibility"] < 0.6:
                    print("  - Focus on citing more credible academic sources")
                if communication_metrics["conversation_helpfulness"] < 0.6:
                    print("  - Enhance response helpfulness and user guidance")
                if research_metrics["research_depth"] < 0.6:
                    print("  - Increase research depth and comprehensive coverage")
                if communication_metrics["citation_quality"] < 0.6:
                    print("  - Improve citation formatting and completeness")
            
            return results
            
        except Exception as e:
            print(f"Error analyzing results: {str(e)}")
            return None
    
    def generate_evaluation_report(self, results, output_file: str = "evaluation_report.json"):
        """Generate a comprehensive evaluation report"""
        try:
            report = {
                "evaluation_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "agent_type": "research_assistant",
                    "evaluation_framework": "langsmith",
                    "dataset_name": "research_assistant_evaluation"
                },
                "performance_summary": {},
                "detailed_results": {},
                "recommendations": []
            }
            
            # Handle structured results data
            if isinstance(results, dict):
                aggregate_scores = results.get('aggregate_scores', {})
                console_output = results.get('console_output', {})
                experiment_name = results.get('experiment_name', 'Unknown')
                
                if aggregate_scores:
                    report["performance_summary"] = dict(aggregate_scores)
                    
                    # Calculate derived metrics
                    research_score = (
                        aggregate_scores.get("research_accuracy", 0) +
                        aggregate_scores.get("research_depth", 0) +
                        aggregate_scores.get("source_credibility", 0)
                    ) / 3
                    
                    communication_score = (
                        aggregate_scores.get("conversation_helpfulness", 0) +
                        aggregate_scores.get("citation_quality", 0)
                    ) / 2
                    
                    report["derived_metrics"] = {
                        "research_quality": research_score,
                        "communication_quality": communication_score,
                        "overall_performance": (research_score + communication_score) / 2
                    }
                
                # Add console output data
                report["experiment_details"] = {
                    "experiment_name": experiment_name,
                    "dataset_name": console_output.get('dataset_name', 'research_assistant_evaluation'),
                    "evaluators_count": console_output.get('evaluators_count', 0),
                    "completion_status": console_output.get('completion_status', 'unknown')
                }
                
            # Handle legacy results format
            elif results and hasattr(results, 'aggregate_scores'):
                report["performance_summary"] = dict(results.aggregate_scores)
                
                # Calculate derived metrics
                scores = results.aggregate_scores
                research_score = (
                    scores.get("research_accuracy", 0) +
                    scores.get("research_depth", 0) +
                    scores.get("source_credibility", 0)
                ) / 3
                
                communication_score = (
                    scores.get("conversation_helpfulness", 0) +
                    scores.get("citation_quality", 0)
                ) / 2
                
                report["derived_metrics"] = {
                    "research_quality": research_score,
                    "communication_quality": communication_score,
                    "overall_performance": (research_score + communication_score) / 2
                }
            
            # Save report
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"Evaluation report saved to: {output_file}")
            return report
            
        except Exception as e:
            print(f"Error generating report: {str(e)}")
            return None

def run_evaluation_experiment():
    """Main function to run the complete evaluation experiment"""
    print("🔬 Starting Research Agent Evaluation Experiment")
    print("=" * 60)
    
    # Check environment variables
    langchain_key = os.getenv("LANGSMITH_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not langchain_key:
        print("❌ LANGSMITH_API_KEY environment variable not set")
        return None
    
    if not openai_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        return None
    
    print("✅ Environment variables configured")
    
    # Initialize evaluator
    evaluator = ResearchAgentEvaluator()
    
    # Run SDK evaluation
    results = evaluator.run_sdk_evaluation()
    
    # Analyze results
    analysis = evaluator.analyze_results(results)
    
    # Generate report
    report = evaluator.generate_evaluation_report(results)
    
    print("\n" + "=" * 60)
    print("EVALUATION EXPERIMENT COMPLETED")
    print("=" * 60)
    
    if results:
        print("✅ Evaluation completed successfully")
        print("📊 Check LangSmith UI for detailed results visualization")
        print("📋 Evaluation report saved locally")
    else:
        print("❌ Evaluation failed - check logs for details")
    
    return {
        "results": results,
        "analysis": analysis,
        "report": report
    }

if __name__ == "__main__":
    # Run the evaluation experiment
    experiment_results = run_evaluation_experiment()
    
    if experiment_results:
        print("\n🎯 Evaluation experiment completed successfully!")
        print("🔗 Visit LangSmith UI to explore detailed results and traces")
    else:
        print("\n❌ Evaluation experiment failed")

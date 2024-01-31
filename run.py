from questionGen import qa_pipeline
from similarityQuestions import relevant_questions
from questionGen import document

# Extract content from the Document object
retrieved_documents = document.content
print("TYPE OF DOCUMENT CONTENT: ", type(document.content))


# Load pre-trained QA mode
def generate_answers_to_questions(questions, documents):
    answers = []

    for question in questions:
        # Ask the model for the answer to the question
        answer = qa_pipeline({
            'question': question,
            'context': documents
        })

        # Append the question, answer, and context to the list
        if 'context' in answer:
            answers.append({
                'question': question,
                'answer': answer['answer'],
                'context': answer['context']
            })
        else:
            answers.append({
                'question': question,
                'answer': answer['answer'],
                # 'context': None  # or provide a default context value
            })

    return answers

# Example usage
generated_questions = relevant_questions
predicted_answers = generate_answers_to_questions(generated_questions, retrieved_documents)

# Display the predicted answers
import json

output_data = {
    "intents": []
}

for qa in predicted_answers:
    intent = {
        "tag": f"intent_{hash(qa['question'])}",  # You can use a better way to generate tags
        "patterns": [qa['question']],
        "responses": [qa['answer']]
    }
    output_data["intents"].append(intent)

# Save to a JSON file
output_file = "my-react-app/src/intents.json"
with open(output_file, 'w', encoding='utf-8') as json_file:
    json.dump(output_data, json_file, indent=2)

print(f"Output data saved to {output_file}")



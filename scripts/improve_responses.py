import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

def analyze_feedback(app_name):
    """Analyze feedback and generate improvement suggestions"""
    feedback_file = f'feedback/{app_name}_feedback.csv'
    if not os.path.exists(feedback_file):
        return
        
    # Load feedback data
    df = pd.read_csv(feedback_file)
    
    # Analyze low-rated responses
    low_rated = df[df['rating'] < 3]
    
    # Group similar queries
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(low_rated['query'])
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Find patterns in problematic responses
    improvements = []
    for idx, row in low_rated.iterrows():
        similar_queries = []
        for i, sim in enumerate(similarity_matrix[idx]):
            if sim > 0.5 and i != idx:
                similar_queries.append(low_rated.iloc[i]['query'])
                
        if similar_queries:
            improvements.append({
                'problem_area': row['query'],
                'similar_queries': similar_queries,
                'comments': row['comment'],
                'frequency': len(similar_queries)
            })
    
    # Generate report
    report_file = f'improvements/{app_name}_improvements.txt'
    os.makedirs('improvements', exist_ok=True)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"Improvement Suggestions for {app_name}\n")
        f.write("=" * 50 + "\n\n")
        
        for imp in sorted(improvements, key=lambda x: x['frequency'], reverse=True):
            f.write(f"Problem Area: {imp['problem_area']}\n")
            f.write(f"Frequency: {imp['frequency']}\n")
            f.write("Similar Queries:\n")
            for query in imp['similar_queries']:
                f.write(f"- {query}\n")
            f.write(f"User Comments: {imp['comments']}\n")
            f.write("-" * 30 + "\n\n")

def update_training_data(app_name):
    """Update training data based on successful responses"""
    feedback_file = f'feedback/{app_name}_feedback.csv'
    if not os.path.exists(feedback_file):
        return
        
    df = pd.read_csv(feedback_file)
    
    # Get highly rated responses
    good_responses = df[df['rating'] > 4]
    
    # Update Q&A pairs
    qa_file = f'documents/{app_name}/learned_responses.txt'
    with open(qa_file, 'a', encoding='utf-8') as f:
        for _, row in good_responses.iterrows():
            f.write(f"\nQ: {row['query']}\n")
            f.write(f"A: {row['response']}\n")

if __name__ == "__main__":
    apps = ['sabi', 'trace', 'katsu']
    for app in apps:
        analyze_feedback(app)
        update_training_data(app) 
from youtube_transcript_api import YouTubeTranscriptApi
import re 
import nltk
from nltk.corpus import stopwords

# Download stop words
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))



# fetch transcript for a video
ytt_api = YouTubeTranscriptApi()
# video id to fetch transcript 
video_id = "5TXf0LDflH8" # Replace with your YouTube video ID to fetch transcript for

# Function to chunk the transcript into smaller parts
def chunk(text, chunk_size = 12, stride = 6):
    words = text.split()
    chunks = []
    
    for i in range(0, len(words) - chunk_size + 1, stride):
        chunk = words[i:i + chunk_size]
        chunks.append(' '.join(chunk))
    
    return chunks
    
    
    
    
    
    
    
# Function to clean the transcript text

def clean(text):
    # Remove URLs
    clean = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove HTML tags
    clean = re.sub(r'<.*?>', '', clean)
    # Remove emojis
    # clean = emoji.replace_emoji(clean, replace='')
    # Split sentences at punctuation marks
    # clean = re.split(r'(?<=[.!?]) +', clean)
    # Remove special characters
    clean = re.sub(r'[^\w\s]', '', clean)
    # Remove timestamps
    clean = re.sub(r'\[\d+:\d+\.\d+s\]', '', clean) 
    # Remove numbers
    #clean = re.sub(r'\d+', '', clean)
    # Convert to lowercase
    clean = clean.lower()
    # Remove stop words
    words = clean.split()
    words = [word for word in words if word not in stop_words]
    clean = ' '.join(words)
    clean  = clean.strip()
    # Remove extra spaces
    clean = re.sub(r'\s+', ' ', clean)
        
    return clean

try:
# fetch transcript
    transcript = ytt_api.fetch(video_id)


# print transcript
#for snippet in transcript:
#   print(snippet.text)

# store results in a file
    all_text = "" 
        
    for snippet in transcript:
        cleaned_text = clean(snippet.text)
        if  len(cleaned_text) > 0:  # Only write non-empty lines
            all_text += ' ' + cleaned_text # Append cleaned text to all_text
    
    chunks = chunk(all_text)
            
    with open("transcript.txt", "w", encoding="utf-8") as f:
        f.write("Transcript\n")
        for chunk in chunks:
            f.write(f"\"{chunk}\"\n")
        
# print("Transcript saved to transcript.txt")
    print("Transcript saved to transcript.txt")
    
# Convert txt to csv
    with open("transcript.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open("transcript.csv", "w", encoding="utf-8") as f:
        f.write("Transcript\n")
        for chunk in chunks:
            f.write(f"\"{chunk}\"\n")

# read and print the transcript from the file
    with open("transcript.txt", "r", encoding="utf-8") as f:
        print(f.read())




except Exception as e:
    print("Error:", str(e))
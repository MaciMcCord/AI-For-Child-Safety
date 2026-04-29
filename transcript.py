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
        piece = words[i:i + chunk_size]
        chunks.append(' '.join(piece))
    
    return chunks
    
     
# Function to clean the transcript text

def clean(text):
    # Remove URLs
    result = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove HTML tags
    result = re.sub(r'<.*?>', '', result)
    # Remove emojis
    # clean = emoji.replace_emoji(clean, replace='')
    # Split sentences at punctuation marks
    # clean = re.split(r'(?<=[.!?]) +', clean)
    # Remove special characters
    result = re.sub(r'[^\w\s]', '', result)
    # Remove timestamps
    result = re.sub(r'\[\d+:\d+\.\d+s\]', '', result) 
    # Remove numbers
    #result = re.sub(r'\d+', '', result)
    # Convert to lowercase
    result = result.lower()
    # Remove stop words
    words = result.split()
    words = [word for word in words if word not in stop_words]
    result = ' '.join(words)
    result  = result.strip()
    # Remove extra spaces
    result = re.sub(r'\s+', ' ', result)
        
    return result

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
        for c in chunks:
            f.write(f"\"{c}\"\n")
        
# print("Transcript saved to transcript.txt")
    print("Transcript saved to transcript.txt")
    
# Convert txt to csv
    with open("transcript.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open("transcript.csv", "w", encoding="utf-8") as f:
        f.write("Transcript\n")
        for c in chunks:
            f.write(f"\"{c}\"\n")

# read and print the transcript from the file
    with open("transcript.txt", "r", encoding="utf-8") as f:
        print(f.read())




except Exception as e:
    print("Error:", str(e))

def get_transcript(video_id: str):
    ytt_api = YouTubeTranscriptApi()
    try:
        transcript = ytt_api.fetch(video_id)
        all_text = ""
        for snippet in transcript:
            cleaned_text = clean(snippet.text)
            if len(cleaned_text) > 0:
                all_text += ' ' + cleaned_text
        return chunk(all_text)
    except Exception as e:
        print("Error:", str(e))
        return []
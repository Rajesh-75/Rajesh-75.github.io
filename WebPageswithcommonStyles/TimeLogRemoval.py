import re

def clean_transcript_file(input_filename, output_filename):
    # Regex Explanation:
    # \d+(?::\d+)+           -> Matches the timestamp (e.g., 0:04, 1:02, 12:37)
    # \s* -> Matches any whitespace
    # (?:                    -> Start of a non-capturing group for duration parts
    #   (?:\d+\s*)?hours?,\s*|  -> Optional: digits + "hour(s),"
    #   (?:\d+\s*)?minutes?,\s*|-> Optional: digits + "minute(s),"
    #   (?:\d+\s*)?seconds?     -> Optional: digits + "second(s)"
    # )+                     -> Repeat to catch multiple units (e.g., minutes AND seconds)
    
    pattern = r'\d+(?::\d+)+\s*(?:(?:\d+\s*)?hours?,\s*|(?:\d+\s*)?minutes?,\s*|(?:\d+\s*)?seconds?)+'

    try:
        with open(input_filename, 'r', encoding='utf-8') as f:
            content = f.read()

        # Remove the matches
        cleaned_text = re.sub(pattern, '', content)

        # Optional: Clean up extra spaces or double newlines left behind
        cleaned_text = re.sub(r' +', ' ', cleaned_text) # Remove double spaces
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(cleaned_text.strip())

        print(f"File cleaned successfully! Results saved to: {output_filename}")

    except FileNotFoundError:
        print(f"Error: The file '{input_filename}' was not found.")

# Run the script
clean_transcript_file('SwamiSarvaPriyaAnd3rd8thverse.txt', 'SwamiSarvaPriyaAnd3rd8thverse2.txt')

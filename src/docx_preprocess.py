from docx import Document
import re

def generate_navigation_dict(document_path):
    document = Document(document_path)

    navigation_dict = {}
    current_heading = None
    current_text = []

    for paragraph in document.paragraphs:
        if paragraph.style.name.startswith('Heading'):
            if current_heading is not None:
                navigation_dict[current_heading] = '\n'.join(current_text)
                current_text = []

            heading_level = int(paragraph.style.name[-1])
            heading_text = paragraph.text
            current_heading = (heading_level, heading_text)
        else:
            current_text.append(paragraph.text)

    if current_heading is not None:
        navigation_dict[current_heading] = '\n'.join(current_text)

    return navigation_dict



def remove_references(text):
    # Regular expression pattern to detect the References section with possible extra lines
    reference_section_pattern = re.compile(
        r'References\s*(?:\n\s*.*?)*(?:\n\s*(\[\d+\].*|TS\s*\d+\.\d+.*))+', re.MULTILINE
    )

    # Search for the reference section pattern
    match = reference_section_pattern.search(text)

    if match:
        # If a match is found, truncate the text from the match start position
        text = text[:match.start()].strip()

    return text


def remove_urls(text):
    # Regular expression pattern to detect URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')

    # Substitute URLs with an empty string
    cleaned_text = re.sub(url_pattern, '', text)
    return cleaned_text



def remove_table_lines(text):
    # Regular expression pattern to detect lines starting with "Table" followed by various formats
    table_line_pattern = re.compile(
        r'^Table\s*[A-Za-z0-9\.\-_]+:\s*.*$', re.MULTILINE
    )

    # Remove table lines
    text = table_line_pattern.sub('', text)

    # Clean up any extra newlines left behind
    text = re.sub(r'\n\s*\n', '\n', text).strip()

    return text


def process_navigation_dict(navigation_dict):

    skip_sections = ['Foreword', 'References']
    processed_dict = {}

    for heading, text in navigation_dict.items():
        if text.strip() and text.strip().lower() != 'void.':
            skip = False
            for section in skip_sections:
                if re.search(r'\b' + re.escape(section) + r'\b', heading[1]):
                    skip = True
                    break
            if not skip:
                # Remove references from the text
                cleaned_text = remove_references(text.strip())
                # Remove table lines from the text
                cleaned_text = remove_table_lines(cleaned_text)
                # Remove URLs from the text
                cleaned_text = remove_urls(cleaned_text)
                processed_dict[heading] = cleaned_text

    return processed_dict


def get_header_chunks(document_path):
    navigation_dict = generate_navigation_dict(document_path)
    processed_dict = process_navigation_dict(navigation_dict)

    return processed_dict

if __name__ == '__main__':

    # Usage example
    document_path = '../data/rel18/38521-3-i11.docx'
    navigation_dict = generate_navigation_dict(document_path)
    processed_dict = process_navigation_dict(navigation_dict)

    for heading, text in processed_dict.items():
        level, heading_text = heading
        print(f"{'  ' * (level - 1)}- {heading_text}")
        print(text)
        print()
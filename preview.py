# Source - https://stackoverflow.com/a/78597183
# Posted by WinnieNotThePooh, modified by community. See post 'Timeline' for change history
# Retrieved 2026-03-24, License - CC BY-SA 4.0

import requests 
from bs4 import BeautifulSoup
    
def fetch_github_repo_info(repo_url):
    # Extract the owner and repo name from the URL
    parts = repo_url.rstrip('/').split('/')
    owner = parts[-2]
    repo = parts[-1]

    # Fetch repository data from GitHub API
    api_url = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {
        'Accept': 'application/vnd.github.v3+json'
    }
    response = requests.get(api_url, headers=headers)
    response.raise_for_status()
    return response.json()

def fetch_github_preview_metadata(repo_url):
    # Fetch the HTML content of the repository page
    response = requests.get(repo_url)
    response.raise_for_status()

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract Open Graph meta tags
    og_title = soup.find('meta', property='og:title')
    og_description = soup.find('meta', property='og:description')
    og_image = soup.find('meta', property='og:image')
    og_url = soup.find('meta', property='og:url')

    # Extract content from the meta tags
    title = og_title['content'] if og_title else 'No title available'
    description = og_description['content'] if og_description else 'No description available'
    image = og_image['content'] if og_image else 'No image available'
    url = og_url['content'] if og_url else repo_url

    return {
        'title': title,
        'description': description,
        'image': image,
        'url': url
    }

def display_preview(metadata, topics):
    print("Title: ", metadata['title'])
    print("Description: ", metadata['description'])
    print("Image URL: ", metadata['image'])
    print("URL: ", metadata['url'])
    print("Topics: ", ', '.join(topics))

if __name__ == '__main__':
    repo_url = input("Enter GitHub repository URL: ")
    repo_info = fetch_github_repo_info(repo_url)
    metadata = fetch_github_preview_metadata(repo_url)
    topics = repo_info.get('topics', [])[:5]  # Get first 5 topics
    display_preview(metadata, topics)

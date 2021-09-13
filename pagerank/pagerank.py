import os
import random
import re
import sys
import copy

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    prob_dist = {}
    num_links = len(corpus[page])

    if num_links:
        for key in corpus:
            prob_dist[key] = (1-damping_factor)/len(corpus)
        for key in corpus[page]:
            prob_dist[key] += damping_factor/num_links
    else:
        for key in corpus:
            prob_dist[key] = 1/len(corpus)
    return prob_dist


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    prob_dist = {}

    for key in corpus:
        prob_dist.update({key: 0})

    my_page = random.choice(list(corpus.keys()))

    for i in range(1, n):
        transition_dist = transition_model(corpus, my_page, damping_factor)
        for key in prob_dist:
            prob_dist[key] = (prob_dist[key]*(i-1) + transition_dist[key])/i  # takes the average of all the values
        my_page = random.choices(list(prob_dist.keys()), list(prob_dist.values()), k=1)[0]  # selects a new random page

    return prob_dist


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    n = len(corpus)

    prob_dist = {}
    for page in corpus:
        prob_dist[page] = 1 / n

    go = True
    while go:
        go = False
        old_dist = copy.deepcopy(prob_dist)
        for page in corpus:
            prob_dist[page] = (1 - damping_factor) / n + damping_factor * sum1(corpus, old_dist, page)
            go = go or abs(old_dist[page] - prob_dist[page]) > 0.001

    return prob_dist


def sum1(corpus, dist, p):
    _sum = 0

    for page in dist:
        if p in corpus[page]:
            _sum += dist[page]/len(corpus[page])

    return _sum


if __name__ == "__main__":
    main()

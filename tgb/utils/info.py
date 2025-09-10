import os.path as osp
import os

r"""
General space to store global information used elsewhere such as url links, evaluation metrics etc.
"""
PROJ_DIR = osp.dirname(osp.abspath(os.path.join(__file__, os.pardir))) + "/"


class BColors:
    """
    A class to change the colors of the strings.
    """

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

DATA_URL_DICT = {
    "tgbl-wiki":"https://object-arbutus.cloud.computecanada.ca/tgb/tgbl-wiki-v2.zip", #"https://object-arbutus.cloud.computecanada.ca/tgb/tgbl-wiki.zip", #v1
    "tgbl-subreddit":"https://object-arbutus.cloud.computecanada.ca/tgb/tgbl-subreddit.zip",
    "tgbl-lastfm":"https://object-arbutus.cloud.computecanada.ca/tgb/tgbl-lastfm.zip",
    "tgbl-review": "https://object-arbutus.cloud.computecanada.ca/tgb/tgbl-review-v2.zip", #"https://object-arbutus.cloud.computecanada.ca/tgb/tgbl-review.zip", #v1
    "tgbl-coin": "https://object-arbutus.cloud.computecanada.ca/tgb/tgbl-coin-v2.zip", #"https://object-arbutus.cloud.computecanada.ca/tgb/tgbl-coin.zip",
    "tgbl-flight": "https://object-arbutus.cloud.computecanada.ca/tgb/tgbl-flight-v2.zip", #"tgbl-flight": "https://object-arbutus.cloud.computecanada.ca/tgb/tgbl-flight_edgelist_v2_ts.zip",
    "tgbl-comment": "https://object-arbutus.cloud.computecanada.ca/tgb/tgbl-comment.zip",
    "tgbn-trade": "https://object-arbutus.cloud.computecanada.ca/tgb/tgbn-trade.zip",
    "tgbn-genre": "https://object-arbutus.cloud.computecanada.ca/tgb/tgbn-genre.zip",
    "tgbn-reddit": "https://object-arbutus.cloud.computecanada.ca/tgb/tgbn-reddit.zip",
    "tgbn-token": "https://object-arbutus.cloud.computecanada.ca/tgb/tgbn-token.zip",
    "tkgl-polecat": "https://object-arbutus.cloud.computecanada.ca/tgb/tkgl-polecat.zip",
    "tkgl-icews": "https://object-arbutus.cloud.computecanada.ca/tgb/tkgl-icews.zip",
    "tkgl-yago":"https://object-arbutus.cloud.computecanada.ca/tgb/tkgl-yago.zip",
    "tkgl-wikidata": "https://object-arbutus.cloud.computecanada.ca/tgb/tkgl-wikidata.zip",
    "tkgl-smallpedia": "https://object-arbutus.cloud.computecanada.ca/tgb/tkgl-smallpedia.zip",
    "thgl-myket": "https://object-arbutus.cloud.computecanada.ca/tgb/thgl-myket.zip",
    "thgl-github": "https://object-arbutus.cloud.computecanada.ca/tgb/thgl-github.zip",
    "thgl-forum": "https://object-arbutus.cloud.computecanada.ca/tgb/thgl-forum.zip",
    "thgl-software": "https://object-arbutus.cloud.computecanada.ca/tgb/thgl-software.zip", #"https://object-arbutus.cloud.computecanada.ca/tgb/thgl-software_ns_random.zip"
    "tkgl-icews14": 'https://drive.google.com/uc?id=1Klc16EY8PEe04S704mGGvLcnNL0E-42B&export=download',# added by counttrucola authors
    "tkgl-icews18": "https://drive.google.com/uc?id=1FPQE014gZ9n-yNicMu4iKosMv-54R17L&export=download", # added by counttrucola authors
    "tkgl-wikiold": "https://drive.google.com/uc?id=1V-oO6usWiVzkyiDRfD87Zj-K9sup1QmP&export=download", # added by counttrucola authors
    "tkgl-gdelt": "https://drive.google.com/uc?id=1QPNJZhu9GqjubCsFdoCt26FFls3X3zfW&export=download", # added by counttrucola authors
    "tkgl-muffi": "https://drive.google.com/uc?id=149SGipvBmLAqUPxl4WzfJ6KT9S0xX123&export=download", # added by counttrucola authors
    "tkgl-mini": "https://drive.google.com/uc?id=149SGipvBmLAqUPxl4WzfJ6KT9S0xX123&export=download", # added by counttrucola authors
    "tkgl-monkey": "https://drive.google.com/uc?id=1SsOEpbUqA_W4O0b2gBhLLHm1xQfo4YTh&export=download" , # added by counttrucola authors
    "tkgl-pig": "https://drive.google.com/uc?id=1JQzTOuiqS93OeWbn9iu4FsSdqJ0QdvUt&export=download", # added by counttrucola authors
    "tkgl-crisis2023": "https://drive.google.com/uc?id=1GqcrUhcUPY6ND7AD7Vgpxt8xR1ctO7vx&export=download", # added by counttrucola authors
} 
DATA_VERSION_DICT = {
    "tgbl-wiki": 2,  
    "tgbl-subreddit": 1,
    "tgbl-lastfm": 1,
    "tgbl-review": 2,
    "tgbl-coin": 2,
    "tgbl-comment": 1,
    "tgbl-flight": 2,
    "tgbn-trade": 1,
    "tgbn-genre": 1,
    "tgbn-reddit": 1,
    "tgbn-token": 1,
    "tkgl-polecat": 1,
    "tkgl-icews": 1,
    "tkgl-yago": 1,
    "tkgl-icews14": 1,
    "tkgl-icews18": 1,
    "tkgl-gdelt": 1,
    "tkgl-wikiold": 1,
    "tkgl-wikidata": 1,
    "tkgl-smallpedia": 1,
    "tkgl-mini": 1,
    "tkgl-monkey": 1,
    "tkgl-pig": 1,
    "tkgl-crisis2023": 1,
    "thgl-myket": 1,
    "thgl-github": 1,
    "thgl-forum": 1,
    "thgl-software": 1,
}


DATA_EVAL_METRIC_DICT = {
    "tgbl-wiki": "mrr",
    "tgbl-subreddit": "mrr",
    "tgbl-lastfm": "mrr",
    "tgbl-review": "mrr",
    "tgbl-coin": "mrr",
    "tgbl-comment": "mrr",
    "tgbl-flight": "mrr",
    "tkgl-polecat": "mrr",
    "tkgl-yago": "mrr",
    "tkgl-icews14": "mrr",
    "tkgl-muffi": "mrr",
    "tkgl-mini": "mrr",
    "tkgl-monkey": "mrr",
    "tkgl-crisis2023": "mrr",
    "tkgl-pig": "mrr",
    "tkgl-icews18": "mrr",
    "tkgl-gdelt": "mrr",
    "tkgl-wikiold": "mrr",
    "tkgl-wikidata": "mrr",
    "tkgl-smallpedia": "mrr",
    "tkgl-icews": "mrr",
    "thgl-myket": "mrr",
    "thgl-github": "mrr",
    "thgl-forum": "mrr",
    "thgl-software": "mrr",
    "tgbn-trade": "ndcg",
    "tgbn-genre": "ndcg",
    "tgbn-reddit": "ndcg",
    "tgbn-token": "ndcg",
}

DATA_NS_STRATEGY_DICT = {
    "tgbl-wiki": "hist_rnd",
    "tgbl-subreddit": "hist_rnd",
    "tgbl-lastfm": "hist_rnd",
    "tgbl-review": "hist_rnd",
    "tgbl-coin": "hist_rnd",
    "tgbl-comment": "hist_rnd",
    "tgbl-flight": "hist_rnd",
    "tkgl-polecat": "time-filtered",
    "tkgl-yago": "time-filtered",
    "tkgl-icews14": "time-filtered",
    "tkgl-icews18": "time-filtered",
    "tkgl-muffi": "time-filtered",
    "tkgl-mini": "time-filtered",
    "tkgl-monkey": "time-filtered",
    "tkgl-crisis2023": "time-filtered",
    "tkgl-pig": "time-filtered",
    "tkgl-gdelt": "time-filtered",
    "tkgl-wikiold": "time-filtered",
    "tkgl-wikidata": "dst-time-filtered",
    "tkgl-smallpedia": "time-filtered",
    "tkgl-icews": "time-filtered",
    "thgl-myket": "node-type-filtered",
    "thgl-github": "node-type-filtered",
    "thgl-forum": "node-type-filtered",
    "thgl-software": "node-type-filtered",
}


DATA_NUM_CLASSES = {
    "tgbn-trade": 255,
    "tgbn-genre": 513,
    "tgbn-reddit": 698,
    "tgbn-token": 1001,
}

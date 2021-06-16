import torch, os, logging, json, random
import numpy as np
# from nltk.corpus import stopwords
from torch.utils.data import Dataset

# stopwords_list = stopwords.words('english')
# stopwords = {i: 1 for i in stopwords_list}
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm


def cal_accuracy(preds, targets):
    return accuracy_score(targets, preds)


def cal_microprecision(preds, targets):
    return precision_score(targets, preds, average="micro")


def cal_microrecall(preds, targets):
    return recall_score(targets, preds, average="micro")


def cal_microf1(preds, targets):
    return f1_score(targets, preds, average='micro')


def cal_macroprecision(preds, targets):
    return precision_score(targets, preds, average="macro")


def cal_macrorecall(preds, targets):
    return recall_score(targets, preds, average="macro")


def cal_macrof1(preds, targets):
    return f1_score(targets, preds, average='macro')


def cal_weightedf1(preds, targets):
    return f1_score(targets, preds, average='weighted')


def cal_weightedprecision(preds, targets):
    return precision_score(targets, preds, average="weighted")


def cal_weightedrecall(preds, targets):
    return recall_score(targets, preds, average="weighted")


def calculate_perf(preds, targets):
    METRICS2FN = {"Accuracy": cal_accuracy,
                  "micro-F1": cal_microf1,
                  "macro-F1": cal_macrof1,
                  "weighted-F1": cal_weightedf1,
                  "macro-Precision": cal_macroprecision,
                  "macro-Recall": cal_macrorecall,
                  "micro-Precision": cal_microprecision,
                  "micro-Recall": cal_microrecall,
                  "weighted-Precision": cal_weightedprecision,
                  "weighted-Recall": cal_weightedrecall}

    return_dict = {}
    for k, v in METRICS2FN.items():
        return_dict[k] = round(v(preds, targets), 4)
    if isinstance(targets[0], list):
        return_dict["support"] = sum([sum(tgt) for tgt in targets])
    else:
        return_dict["support"] = len(targets)
    return return_dict


def read_jsonl(filepath):
    examples = []
    with open(filepath, "r") as f:
        for line in f:
            example = json.loads(line.strip())
            examples.append(example)
    return examples

def get_short_cates(cates):
    short_cates = []
    for each in cates.split(","):
        short_cates.append(each.split("-")[-1])
    return ",".join(short_cates)

def add_filehandler_for_logger(output_path, logger, out_name="log"):
    logFormatter = logging.Formatter('%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s')
    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)
    fileHandler = logging.FileHandler(os.path.join(output_path, f"{out_name}.txt"), mode="a")
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_class_names(filepath):
    classes = []
    with open(filepath, "r") as f:
        for line in f:
            classes.append(line.strip())
    return classes


class MyDataset(Dataset):
    def __init__(self, encoded_examples):
        self.encoded_examples = encoded_examples

    def __getitem__(self, index):
        selected_to_return = {}
        for k, v in self.encoded_examples.items():
            selected_to_return[k] = v[index]
        return selected_to_return

    def __len__(self):
        return len(self.encoded_examples["input_ids"])



priority2numeric = {"Critical": 1.0, "High": 0.75, "Medium": 0.5, "Low": 0.25}
# this is strict as specified by the guidelines:
# the classes_categories.txt in data/ should also follow this order strictly
its_list = ['CallToAction-Donations',
            'CallToAction-MovePeople',
            'CallToAction-Volunteer',
            'Other-Advice',
            'Other-ContextualInformation',
            'Other-Discussion',
            'Other-Irrelevant',
            'Other-Sentiment',
            'Report-CleanUp',
            'Report-EmergingThreats',  #
            'Report-Factoid',
            'Report-FirstPartyObservation',
            'Report-Hashtags',  #
            'Report-Location',  #
            'Report-MultimediaShare',  #
            'Report-News',  #
            'Report-NewSubEvent',  #
            'Report-Official',
            'Report-OriginalEvent',
            'Report-ServiceAvailable',
            'Report-ThirdPartyObservation',  #
            'Report-Weather',
            'Request-GoodsServices',
            'Request-InformationWanted',
            'Request-SearchAndRescue']  # order alphabetically, this starts being used since 2020b
its_list = [i.split("-")[-1] for i in its_list]

event2test_2019b = {"albertaWildfires2019": "TRECIS-CTIT-H-Test-029",
                    "cycloneKenneth2019": "TRECIS-CTIT-H-Test-030",
                    "philippinesEarthquake2019": "TRECIS-CTIT-H-Test-031",
                    "coloradoStemShooting2019": "TRECIS-CTIT-H-Test-032",
                    "southAfricaFloods2019": "TRECIS-CTIT-H-Test-033",
                    "sandiegoSynagogueShooting2019": "TRECIS-CTIT-H-Test-034"}

event2test_2020at1t2 = {'athensEarthquake2020': 'TRECIS-CTIT-H-Test-35',
                        'baltimoreFlashFlood2020': 'TRECIS-CTIT-H-Test-36',
                        'brooklynBlockPartyShooting2020': 'TRECIS-CTIT-H-Test-37',
                        'daytonOhioShooting2020': 'TRECIS-CTIT-H-Test-38',
                        'elPasoWalmartShooting2020': 'TRECIS-CTIT-H-Test-39',
                        'gilroygarlicShooting2020': 'TRECIS-CTIT-H-Test-40',
                        'hurricaneBarry2020': 'TRECIS-CTIT-H-Test-41',
                        'indonesiaEarthquake2020': 'TRECIS-CTIT-H-Test-42',
                        'keralaFloods2020': 'TRECIS-CTIT-H-Test-43',
                        'myanmarFloods2020': 'TRECIS-CTIT-H-Test-44',
                        'papuaNewguineaEarthquake2020': 'TRECIS-CTIT-H-Test-45',
                        'siberianWildfires2020': 'TRECIS-CTIT-H-Test-46',
                        'typhoonKrosa2020': 'TRECIS-CTIT-H-Test-47',
                        'typhoonLekima2020': 'TRECIS-CTIT-H-Test-48',
                        'whaleyBridgeCollapse2020': 'TRECIS-CTIT-H-Test-49'}

event2test_2020at3 = {'covidDC2020': 'TRECIS-CTIT-H-Test-50', 'covidNYC2020': 'TRECIS-CTIT-H-Test-51',
                      'covidWashingtonState2020': 'TRECIS-CTIT-H-Test-52'}

event2test_2020bt1t2 = {'houstonExplosion2020': '53', 'texasAMCommerceShooting2020': '54',
                        'southeastTornadoOutbreak2020': '55', 'stormCiara2020': '56', 'stormDennis2020': '57',
                        'portervilleLibraryFire2020': '58', 'virraMallHostageSituation2020': '59',
                        'stormJorge2020': '60',
                        'tennesseeTornadoOutbreak2020': '61', 'tennesseeDerecho2020': '62',
                        'edenvilleDamFailure2020': '63', 'sanFranciscoPierFire2020': '64',
                        'tropicalStormCristobal2020': '65', 'beirutExplosion2020': '66'}

event2test_2020bt3 = {'covidMiami2020': '67', 'covidJacksonville2020': '68', 'covidHouston2020': '69',
                      'covidPhoenix2020': '70', 'covidAtlanta2020': '71', 'covidNYC2020': '72',
                      'covidSeattle2020': '73', 'covidMelbourne2020': '74', 'covidNewZealand2020': '75'}

task_editions_available = ["2019b", "2020at1", "2020at2", "2020at3", "2020bt1", "2020bt2", "2020bt3", "2021a"]
edition2eventmapping = {"2019b": event2test_2019b, "2020at1": event2test_2020at1t2, "2020at2": event2test_2020at1t2,
                        "2020at3": event2test_2020at3, "2020bt1": event2test_2020bt1t2, "2020bt2": event2test_2020bt1t2,
                        "2020bt3": event2test_2020bt3}

for key, mappings in edition2eventmapping.items():
    edition2eventmapping[key] = {event_id.lower(): event_label for event_id, event_label in mappings.items()}


def get_normalized_it2priorityscore():
    # to get it2priorityscore: analyze the training set
    it2priorityscore = {'Advice': 0.22862577231414025, 'CleanUp': 0.35653795010024114,
                        'News': 0.3270722902848146, 'Discussion': 0.19307661376501264,
                        'Donations': 0.3612767541359874, 'EmergingThreats': 0.8537984236364585,
                        'Factoid': 0.48917308056907677, 'FirstPartyObservation': 0.1420197717109446,
                        'GoodsServices': 0.6959346525995092, 'Hashtags': 0.2687451987058043,
                        'InformationWanted': 0.6688596972384949, 'Irrelevant': 0.0010061402766415601,
                        'OriginalEvent': 0.09192532381788204, 'MovePeople': 0.9253136744180216,
                        'MultimediaShare': 0.3128463518966703, 'Official': 0.6785618345463643,
                        'ContextualInformation': 0.1405539758609653, 'SearchAndRescue': 1.0010061402766415,
                        'Sentiment': 0.039519645594885515, 'ServiceAvailable': 0.7669372401298534,
                        'NewSubEvent': 0.9176682650090996, 'ThirdPartyObservation': 0.2703955630411556,
                        'Location': 0.03711651480530716, 'Volunteer': 0.3839639462617982,
                        'Weather': 0.33196659834790165}
    scores = list(it2priorityscore.values())
    max = np.max(scores)
    min = np.min(scores)
    nomalized = {}
    for key, score in it2priorityscore.items():
        nomalized[key] = (score - min) / (max - min)
    return nomalized


def submit_ensemble(src_runs, load_path, edition="2020bt1", runtag="default"):
    edition = "2020bt1" if edition == "2020b" else edition
    runs_list = []
    for src_run in src_runs:
        lines = {}
        with open(os.path.join(load_path, src_run), "r") as f:
            for line in tqdm(f,desc="reading"):
                if edition!="2021a":
                    columns = line.strip().split("\t")
                    event_id = columns[0]
                    post_id = columns[2]
                    priority = columns[4]
                    its_scores = columns[5]
                    lines[post_id] = {"event_id": event_id, "post_id": post_id, "priority": priority, "its_scores": json.loads(its_scores)}
                else:
                    ins = json.loads(line.strip())
                    event_id = ins["topic"]
                    post_id = ins["tweet_id"]
                    priority = ins["priority"]
                    its_scores = ins["info_type_scores"]
                    lines[post_id] = {"event_id": event_id, "post_id": post_id, "priority": priority, "its_scores": its_scores}
        runs_list.append(lines)

    reference = runs_list[0]
    combined_lines = {}

    for post_id, line in tqdm(reference.items(),desc="combining"):
        individual_priorities = [line["priority"]]
        post_id = line["post_id"]
        event_id = line["event_id"]
        individual_it_scores = [line["its_scores"]]

        for other_run in runs_list[1:]:
            if event_id == other_run[post_id]["event_id"]:
                pri = other_run[post_id]["priority"]
                individual_priorities.append(pri)
                individual_it_scores.append(other_run[post_id]["its_scores"])

        # get the highest of individual predictions for priority
        individual_priorities = [float(i) for i in individual_priorities]
        priority = max(individual_priorities)
        # priority = sum(individual_priorities)/len(individual_priorities)

        # get the combination of individual predictions for information types
        its_scores = np.array(individual_it_scores).max(0).tolist()
        its_labels = [1 if each > 0.5 else 0 for each in its_scores]

        if its_labels[its_list.index("Irrelevant")] == 1 and sum(its_labels) > 1:
            its_scores[its_list.index("Irrelevant")] = 0.0
            its_labels[its_list.index("Irrelevant")] = 0

        if event_id not in combined_lines:
            combined_lines[event_id] = []
        combined_lines[event_id].append([event_id, post_id, its_scores, its_labels, priority])

    sub_path = os.path.join(load_path, runtag)
    with open(sub_path, "w+") as f:
        for event_id, lines in tqdm(combined_lines.items(),desc="writing"):
            ranked_lines = list(sorted(lines, key=lambda k: k[-1], reverse=True))
            rank = 1
            for line in ranked_lines:
                if edition != "2021a":
                    line_sub = line[0] + "\tQ0\t" + str(line[1]) + "\t" + str(rank) + "\t" + str(line[-1]) + "\t" + json.dumps(
                        line[2]) + "\t" + runtag + "\n"
                    f.write(line_sub)
                    rank += 1
                else:
                    line_sub=json.dumps({"topic": line[0], "runtag": runtag, "tweet_id": str(line[1]), "priority": line[-1], "info_type_scores": line[2], "info_type_labels": line[3]})+"\n"
                    # line_sub = line_sub.replace(", ",",")
                    f.write(line_sub)

    # gzip submission
    import gzip
    print("gzip...")
    f_in = open(sub_path, "rb")
    gzip_submit_path = os.path.join(load_path, runtag + ".gz")
    f_out = gzip.open(gzip_submit_path, 'wb')
    f_out.write(f_in.read())
    f_out.close()
    f_in.close()
    print("Find the generated submission in: ", sub_path)
    print("Find the generated submission (gzipped) in: ", gzip_submit_path)

# def normalize_priority_preds(scores):
#     max = np.max(scores)
#     min = np.min(scores)
#     return [(score - min) / (max - min) for score in scores]

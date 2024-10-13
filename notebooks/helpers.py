import pandas as pd
import hashlib
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import ollama
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List
from enum import Enum
import re
import string
from tqdm.notebook import tqdm
import ast
import os
import glob
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pykeen
from pykeen import predict
from pykeen.datasets import Nations
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pathlib import Path
import itertools
import torch
import warnings
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import seaborn as sns
import random
from datetime import datetime
import itertools

# Filter warnings
warnings.filterwarnings('ignore')


###### OLLAMA #####

class Domain(str, Enum):
    ENVIRONMENT = "ENVIRONMENT"
    CITIES = "CITIES"
    LOGISTICS = "LOGISTICS"
    AGRIFOOD = "AGRIFOOD"
    SENSORING = "SENSORING"
    ROBOTICS = "ROBOTICS"
    WATER = "WATER"
    AERONAUTICS = "AERONAUTICS"
    HEALTH = "HEALTH"
    ENERGY = "ENERGY"
    DESTINATION = "DESTINATION"
    MANUFACTURING = "MANUFACTURING"
    AUTOMOTIVE = "AUTOMOTIVE"
    TRANSPORTATION = "TRANSPORTATION"
    FINANCE = "FINANCE"
    EDUCATION = "EDUCATION"
    TELECOMMUNICATIONS = "TELECOMMUNICATIONS"

# enables `response_model` in create call
client = instructor.from_openai(
    OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # required, but unused
    ),
    mode=instructor.Mode.JSON,
)

def ollama_response(model, content, prompt):

    resp = client.chat.completions.create(
        model=model,
        response_model=Domain,
        messages=[
            {
                "role": "user",
                "content": f"{prompt}: '{content}'",
            },
        ],
    )

    return resp

###### GENERAL ######

def create_dict_from_df(df):
    dcat3_dict = {}
    dcat3_classes = df[df['property'] == 'CLASS']["class"].values
    dcat3_properties = []
    for c in dcat3_classes:
        dcat3_dict[c] = {
            'properties': []
        }
        vals = df[df['class'] == c][['property', 'definition', 'range', 'rdf_property']].values
        for val in vals:
            if val[0] == 'CLASS':
                continue
            dcat3_dict[c]['properties'].append({
                'property': val[0],
                'definition': val[1],
                'range': val[2],
                'rdf_property': val[3],
            })
            dcat3_properties.append(val[0])
        
    return dcat3_dict, dcat3_classes, dcat3_properties

def encode_string(input):
    s = input.encode()
    return hashlib.sha256(s).hexdigest()

def find_id(search_key, chunck_id_mapper):
    # Schlüssel finden
    found_key = None
    for key, value in chunck_id_mapper.items():
        if search_key in value:
            found_key = key
            break
    
    return found_key

##### Named Entity Recognition #####
# Konsolidierte Liste von Named Entity Typen
entity_types = [
    "PERSON",          # Personenname
    "NORP",            # Nationalitäten, religiöse und politische Gruppen
    "FAC",             # Gebäude, Straßen, Denkmäler
    "ORG",             # Organisationen
    "GPE",             # Geopolitische Einheiten (Länder, Städte)
    "LOC",             # Geografische Orte
    "PRODUCT",         # Produkte und Objekte
    "EVENT",           # Ereignisse
    "WORK_OF_ART",     # Kunstwerke (Bücher, Filme, etc.)
    "LAW",             # Gesetzestexte und Verträge
    "LANGUAGE",        # Sprachen
    "DATE",            # Datumsangaben
    "TIME",            # Zeitangaben
    "PERCENT",         # Prozentzahlen
    "MONEY",           # Geldbeträge
    "QUANTITY",        # Mengenangaben
    "ORDINAL",         # Ordinalzahlen
    "CARDINAL",        # Kardinalzahlen
    "MEDICATION",      # Medikamente (erweitert)
    "DISEASE",         # Krankheiten (erweitert)
    "ORGANIZATION_UNIT", # Abteilungen in Unternehmen/Organisationen (erweitert)
    "TECHNOLOGY",      # Technologische Begriffe (erweitert)
    "SOFTWARE",        # Software-Produkte (erweitert)
    "HARDWARE"         # Hardware-Komponenten (erweitert)
]

# Liste von Prädikaten (Kanten) für die Entitäten
entity_edges = [
    "is_member_of",    # Verbindung: PERSON -> ORG, NORP
    "is_part_of",      # Verbindung: FAC -> GPE, ORG
    "is_located_in",   # Verbindung: LOC, ORG, FAC -> GPE, LOC
    "has_product",     # Verbindung: ORG -> PRODUCT, SOFTWARE, HARDWARE
    "hosts_event",     # Verbindung: LOC, ORG -> EVENT
    "authored_by",     # Verbindung: WORK_OF_ART -> PERSON, ORG
    "enacts",          # Verbindung: LAW -> ORG, GPE
    "speaks",          # Verbindung: PERSON, ORG -> LANGUAGE
    "occurs_on",       # Verbindung: EVENT -> DATE, TIME
    "is_part_of_group",# Verbindung: PERSON -> ORGANIZATION_UNIT
    "uses_technology", # Verbindung: PERSON, ORG -> TECHNOLOGY
    "treats",          # Verbindung: MEDICATION -> DISEASE
    "costs",           # Verbindung: PRODUCT, SERVICE -> MONEY
    "has_quantity_of", # Verbindung: PRODUCT -> QUANTITY
    "is_first_of",     # Verbindung: ORDINAL -> ENTITY
    "counts",          # Verbindung: CARDINAL -> ENTITY
    "produces",        # Verbindung: ORG -> PRODUCT, SOFTWARE, HARDWARE
    "is_version_of",   # Verbindung: SOFTWARE -> SOFTWARE (Versionen)
    "belongs_to",      # Verbindung: ENTITY -> ORG, GPE (Zugehörigkeit)
    "represents",      # Verbindung: NORP -> GPE
]

###### EXTRACTION ########

def triple_extract(text, entity_types=entity_types, entity_edges=entity_edges):

    input_format = """Perform Named Entity Recognition (NER) and extract knowledge graph triplets from the text. NER identifies named entities of given entity types, and triple extraction identifies relationships between entities using specified predicates.

        **Entity Types:**
        {entity_types}

        **Predicates:**
        {entity_edges}

        **Text:**
        {text}
        """

    message = input_format.format(
                entity_types = json.dumps({"entity_types": entity_types}),
                entity_edges = json.dumps({"predicates": entity_edges}),
                text = text)

    messages = [{'role': 'user', 'content': message}]
    return messages


domains = ["CITIES", "ENVIRONMENT", "CROSS SECTOR", "LOGISTICS", "AGRIFOOD", "SENSORING", "ROBOTICS", "WATER", "AERONAUTICS", "HEALTH", "ENERGY", "DESTINATION", "MANUFACTURING"]

def domain_extract(description, domains=domains):

    input_format = """Categorize the following dataset description into one of the predefined categories. You should select the most appropriate category from the given list based on the content of the description. If you are uncertain or the description spans multiple domains, use the "CROSS_SECTOR" category. Only Return the name of the Domain - nothing else.

        **Entity Types:**
        {domains}

        **Text:**
        {description}
        """

    message = input_format.format(
                domains = json.dumps({"domains": domains}),
                description = description)

    messages = [{'role': 'user', 'content': message}]
    return messages

def parse_triples(triples):

    entities = []
    triples = []

    # Die Entitäten und Tripel iterieren
    for item in triples["entities_and_triples"]:
        # Wenn es sich um eine Entität handelt (erkennbar am Doppelpunkt), dann extrahieren
        if ':' in item:
            match = re.match(r"\[(\d+)\],\s+(\w+):(.+)", item)
            if match:
                entity_id = match.group(1)
                entity_type = match.group(2)
                entity_value = match.group(3).strip()
                entities.append({"id": int(entity_id), "type": entity_type, "value": entity_value})
        # Wenn es sich um ein Tripel handelt, erkennbar am Muster "[id1] RELATION [id2]"
        else:
            match = re.match(r"\[(\d+)\]\s+(\w+)\s+\[(\d+)\]", item)
            if match:
                subject_id = match.group(1)
                relation = match.group(2)
                object_id = match.group(3)
                triples.append({"subject": subject_id, "relation": relation, "object": object_id})

    # Ausgabe
    result = {
        "entities": entities,
        "triples": triples
    }

    return result

def create_splitter():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False,
    )

    return splitter

def clean_text(text):
    # Interpunktion entfernen
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Erst Leerzeichen durch Unterstriche ersetzen
    text = text.replace(" ", "_")
    
    # Alle sonstigen Sonderzeichen, die du entfernen möchtest, kannst du optional ergänzen
    return text.lower()

palette = "hls"

## Now add these colors to communities and make another dataframe
def colors2Community(communities, nodes) -> pd.DataFrame:
    ## Define a color palette
    p = sns.color_palette(palette, len(communities)).as_hex()
    random.shuffle(p)
    rows = []
    group = 0
    for community in communities:
        color = p.pop()
        group += 1
        for node in community:
            rows += [{"node": node, "color": color, "group": nodes[node]['type'], "ids": nodes[node]['id']}]
    df_colors = pd.DataFrame(rows)
    return df_colors
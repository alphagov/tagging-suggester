# This script presents suggestions for untagging or retagging from `archive/5_regression_retag_content_business_tax.py`
# for usage in govuk-interactive-form.
# Move the output from `5_regression_retag_content_business_tax.py` into a directory below the current one, called
# `for_review` and run this script.

import csv
import yaml
import urllib.request, json
from os import listdir
import pry

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

def get_parent_taxons(taxon):
    possible_parent_taxons = taxon.get("links", {}).get("parent_taxons", {})
    if any(possible_parent_taxons):
        for parent_taxon in possible_parent_taxons:
            while parent_taxon != {}:
                possible_parent_taxon = parent_taxon.get("links", {}).get("parent_taxons", {})
                if possible_parent_taxon != {}:
                    parent_taxon = possible_parent_taxon[0]
                else:
                    break
            return [parent_taxon]
    else:
        return [""]

def get_all_parent_taxons(content_item, field):
    taxons = content_item.get("links", {}).get(field, {})
    all_parent_taxons = []
    for taxon in taxons:
        all_parent_taxons.append([taxon] + get_parent_taxons(taxon))
    return all_parent_taxons

def get_formatted_taxon_names(content_item, field = "taxons", current_taxon_title = None):
    taxon_names = get_all_parent_taxons(content_item, field)
    formatted_names = ""
    if any(taxon_names):
        for taxons in taxon_names:
            taxons.reverse()
            taxons = [ taxon['title'] if type(taxon) is dict else '' for taxon in taxons ]
            if current_taxon_title is not None:
                taxons.append(current_taxon_title)
            joined_names = " > ".join(taxons)
            formatted_names += remove_prefix(joined_names, " > ")
        return formatted_names
    else:
        if current_taxon_title is not None:
            return current_taxon_title
        else:
            "None"

def get_taxon_parents(taxon_base_path):
    with urllib.request.urlopen("http://www.gov.uk/api/content/" + taxon_base_path) as url:
        content_item = json.loads(url.read().decode())
        return get_all_parent_taxons(content_item, 'parent_taxons')

def get_taxon_name_path(taxon_base_path):
    with urllib.request.urlopen("http://www.gov.uk/api/content/" + taxon_base_path) as url:
        content_item = json.loads(url.read().decode())
        return get_formatted_taxon_names(content_item, 'parent_taxons', content_item["title"])

def get_content(retagging_info, content_item):
    title = content_item["title"]
    description = content_item["description"]
    full_url = "https://www.gov.uk" + content_item["base_path"]
    return f"<h2><a href='{full_url}' target='_blank'>{title}</a></h2><p>{description}</p>"

def get_retagging_answers(retagging_info, content_item):
    answers = []
    text = f"<a href='https://www.gov.uk{retagging_info['current_taxon_base_path']}' target='_blank'>{get_taxon_name_path(retagging_info['current_taxon_base_path'])}</a>"
    current_tagging  = { 'key': retagging_info["current_taxon_base_path"], 'text': text }
    answers.append(current_tagging)
    for parent_taxons in get_taxon_parents(retagging_info["suggestion_base_path"]):
        for taxon in parent_taxons:
            if type(taxon) is dict:
                answers = conditionally_append_taxon_to_answer_list(answers, taxon)
                for taxon_parent in get_taxon_parents(taxon["base_path"]):
                    for taxon_parent_taxon in taxon_parent:
                        if type(taxon_parent_taxon) is dict:
                            answers = conditionally_append_taxon_to_answer_list(answers, taxon_parent_taxon)
    answers = sorted(answers, key=lambda answer: answer['text'].count(">"))
    answers.append({ 'key': 'none', 'text': 'None of the above' })
    return answers

def conditionally_append_taxon_to_answer_list(answers, taxon):
    if any(answer['key'] == taxon['base_path'] for answer in answers) == False:
        text = f"<a href='https://www.gov.uk{taxon['base_path']}' target='_blank'>{get_taxon_name_path(taxon['base_path'])}</a>"
        suggested_tag = { 'key': taxon['base_path'], 'text': text }
        answers.append(suggested_tag)
    return answers

dir_name = "for_review"
output_filenames = []
for filename in listdir(dir_name):
    print("Presenting: " + filename)
    filename_to_open = dir_name + "/" + filename
    if "retag" in filename_to_open:
        output = {}
        output["questions"] = []
        with open(filename_to_open, 'r') as csvfile:
            content_to_retag = csv.DictReader(csvfile)
            for retagging_info in content_to_retag:
                retag_output = {}
                with urllib.request.urlopen("http://www.gov.uk/api/content/" + retagging_info["content_to_retag_base_path"]) as url:
                    content_item = json.loads(url.read().decode())
                    retag_output["id"] = retagging_info["content_to_retag_base_path"]
                    retag_output["question"] = "Which topic should the following page be tagged to?"
                    retag_output["content"] = get_content(retagging_info, content_item)
                    retag_output["url"] = ""
                    retag_output["answers"] = get_retagging_answers(retagging_info, content_item)
                    retag_output["more_detail_prompt"] = "Can you suggest a different topic or provide more detail?"
                    retag_output["key_to_show_more_detail_prompt"] = "none"
                    unique_name = retagging_info["content_to_retag_base_path"] + retagging_info["current_taxon_base_path"]
                    if unique_name not in output["questions"]:
                        output['questions'].append(retag_output)
        if any(output):
            output_filename = filename.split(".csv")[0] + "_items.yml"
            output_filenames.append(output_filename)
            with open(output_filename, 'w') as output_file:
                yaml.dump(output, output_file, default_flow_style=False)
    if "untag" in filename_to_open:
        output = []
        already_appended = []
        with open('depth_first_content_to_untag_Money.csv', 'r') as csvfile:
            content_to_untag = csv.DictReader(csvfile)
            for untagging_info in content_to_untag:
                retag_output = {}
                with urllib.request.urlopen("http://www.gov.uk/api/content/" + untagging_info["content_to_retag_base_path"]) as url:
                    content_item = json.loads(url.read().decode())
                    retag_output["id"] = untagging_info["content_to_retag_base_path"]
                    retag_output["question"] = "Should this page be untagged?"
                    retag_output["content"] = get_content(untagging_info, content_item)
                    retag_output["url"] = ""
                    unique_name = untagging_info["content_to_retag_base_path"] + untagging_info["current_taxon_base_path"]
                    if unique_name not in already_appended:
                        already_appended.append(unique_name)
                        output.append(retag_output)
        if any(output):
            output_filename = filename.split(".yml")[0] + "_items.csv"
            output_filenames.append(output_filename)
            with open(output_filename, 'w') as output_file:
                yaml.dump(output, output_file, default_flow_style=False)
for output_filename in output_filenames:
    print(output_filename)
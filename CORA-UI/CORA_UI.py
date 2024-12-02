from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QListWidgetItem
import Feature_Extraction as FE
import pandas as pd
import requests

def call_api(feature_vector, citations, api_url="http://127.0.0.1:5000/predict"):
    """
    Calls the prediction API with a given feature vector and citations.

    Args:
        feature_vector (list): The feature vector of the new node.
        citations (list): List of node IDs (connections) for the new node.
        api_url (str): The URL of the API endpoint.

    Returns:
        dict: The API response containing predictions or an error message.
    """
    payload = {
        "feature_vector": feature_vector,
        "connections": citations
    }
    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def map_citations_to_legitimate_titles_from_csv(citations, csv_file_path):
    """
    Maps the citation fake names to their legitimate titles using a CSV file.
    """
    nodes_with_legitimate_titles_df = pd.read_csv(csv_file_path)
    fake_name_to_legit_title = dict(zip(nodes_with_legitimate_titles_df["Fake Name"], nodes_with_legitimate_titles_df["Top Paper Title"]))
    return {citation: fake_name_to_legit_title.get(citation, "Unknown Title") for citation in citations}


def log_message(message, color="white"):
    """Add a message to the log box."""
    dlg.log_box.setTextColor(QColor(color))
    dlg.log_box.append(message)

def save_paper():
    """Save the paper abstract."""
    abstract = dlg.textEdit.toPlainText().strip()
    if abstract:
        dlg.textEdit.setPlainText("")
        log_message("Paper Abstract: " + abstract, color="blue")
    else:
        log_message("EMPTY STRING ENTERED", color="red")

def clear_paper():
    """Clear the paper abstract."""
    dlg.textEdit.setPlainText("")
    log_message("Paper Abstract cleared.", color="blue")
    log_message("Try Again: ", color="blue")

def save_citations(legit_citations):
    """Save selected, checked, and entered citations."""
    saved_citations = []
    for i in range(dlg.listWidget.count()):
        item = dlg.listWidget.item(i)
        if item.checkState() == Qt.Checked:
            title = item.text()
            for key, value in legit_citations.items():
                if value == title:
                    paper_id = key.split("_")[-1]
                    saved_citations.append(int(paper_id))
                    break
    entered_text = dlg.lineEdit_2.text().strip()
    if entered_text:
        for key, value in legit_citations.items():
            if entered_text.lower() == value.lower():
                paper_id = key.split("_")[-1]
                if int(paper_id) not in saved_citations:
                    saved_citations.append(int(paper_id))
                break
    if saved_citations:
        log_message("Citations saved successfully!", color="blue")
        log_message(str(saved_citations))
    else:
        log_message("INVALID Citations!", color="red")
    return saved_citations

def mark_searched_item(text):
    """Mark the searched item in the QListWidget."""
    for i in range(dlg.listWidget.count()):
        item = dlg.listWidget.item(i)
        if text.lower() in item.text().lower():
            item.setSelected(True)
        else:
            item.setSelected(False)

def make_api_call():
    """Process the abstract and call the API."""
    abstract = dlg.textEdit.toPlainText().strip()
    feature_vector = FE.process_abstract(abstract)
    saved_citations = save_citations(legit_citations)
    api_url = "http://127.0.0.1:5000/predict"
    response = call_api(feature_vector, saved_citations, api_url)
    if "predictions" in response:
        log_message("Predictions: " + str(response["predictions"]), color="green")
    else:
        log_message("Error: " + response.get("error", "Unknown error occurred"), color="red")

def main():
    app = QtWidgets.QApplication([])
    global dlg
    dlg = uic.loadUi(r"C:\\Users\\mersh\\OneDrive\\Desktop\\CoraProject\\CORA-UI\\CORA.ui")
    
    dlg.textEdit.setPlaceholderText("Enter Paper Abstract")
    dlg.lineEdit_2.setPlaceholderText("Search for a citation...")
    dlg.log_box.setReadOnly(True)
    dlg.log_box.setPlaceholderText("Log messages will appear here...")
    dlg.listWidget.setSelectionMode(QtWidgets.QListWidget.MultiSelection)
    citations=[
    "Paper_CaseBased_1420",
    "Paper_CaseBased_2415",
    "Paper_CaseBased_2338",
    "Paper_CaseBased_66",
    "Paper_CaseBased_785",
    "Paper_CaseBased_2122",
    "Paper_CaseBased_922",
    "Paper_CaseBased_1033",
    "Paper_CaseBased_49",
    "Paper_CaseBased_1535",
    "Paper_GeneticAlgorithms_2152",
    "Paper_GeneticAlgorithms_2173",
    "Paper_GeneticAlgorithms_2512",
    "Paper_GeneticAlgorithms_721",
    "Paper_GeneticAlgorithms_1230",
    "Paper_GeneticAlgorithms_357",
    "Paper_GeneticAlgorithms_803",
    "Paper_GeneticAlgorithms_880",
    "Paper_GeneticAlgorithms_1784",
    "Paper_GeneticAlgorithms_2704",
    "Paper_NeuralNetworks_263",
    "Paper_NeuralNetworks_1373",
    "Paper_NeuralNetworks_2631",
    "Paper_NeuralNetworks_1612",
    "Paper_NeuralNetworks_2671",
    "Paper_NeuralNetworks_1751",
    "Paper_NeuralNetworks_1947",
    "Paper_NeuralNetworks_2540",
    "Paper_NeuralNetworks_589",
    "Paper_NeuralNetworks_1766",
    "Paper_ProbabilisticMethods_2677",
    "Paper_ProbabilisticMethods_485",
    "Paper_ProbabilisticMethods_1147",
    "Paper_ProbabilisticMethods_170",
    "Paper_ProbabilisticMethods_2323",
    "Paper_ProbabilisticMethods_2266",
    "Paper_ProbabilisticMethods_976",
    "Paper_ProbabilisticMethods_973",
    "Paper_ProbabilisticMethods_2660",
    "Paper_ProbabilisticMethods_993",
    "Paper_ReinforcementLearning_2480",
    "Paper_ReinforcementLearning_2655",
    "Paper_ReinforcementLearning_1529",
    "Paper_ReinforcementLearning_2007",
    "Paper_ReinforcementLearning_85",
    "Paper_ReinforcementLearning_879",
    "Paper_ReinforcementLearning_2118",
    "Paper_ReinforcementLearning_562",
    "Paper_ReinforcementLearning_2485",
    "Paper_ReinforcementLearning_1540",
    "Paper_RuleLearning_249",
    "Paper_RuleLearning_654",
    "Paper_RuleLearning_2312",
    "Paper_RuleLearning_1271",
    "Paper_RuleLearning_2217",
    "Paper_RuleLearning_196",
    "Paper_RuleLearning_316",
    "Paper_RuleLearning_1165",
    "Paper_RuleLearning_1781",
    "Paper_RuleLearning_1919",
    "Paper_Theory_1181",
    "Paper_Theory_2653",
    "Paper_Theory_2391",
    "Paper_Theory_1712",
    "Paper_Theory_1173",
    "Paper_Theory_2059",
    "Paper_Theory_2329",
    "Paper_Theory_1671",
    "Paper_Theory_2080",
    "Paper_Theory_1339"
]
    
    global legit_citations
    legit_citations = map_citations_to_legitimate_titles_from_csv(
        citations, r"C:\\Users\\mersh\\OneDrive\\Desktop\\CoraProject\\CORA-UI\\Nodes_with_Legitimate_Titles.csv"
    )
    for citation, title in legit_citations.items():
        item = QListWidgetItem(title)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Unchecked)
        dlg.listWidget.addItem(item)

    # Connect buttons to actions
    dlg.pushButton_2.clicked.connect(save_paper)
    dlg.pushButton.clicked.connect(clear_paper)
    dlg.pushButton_3.clicked.connect(make_api_call)
    dlg.lineEdit_2.textChanged.connect(lambda: mark_searched_item(dlg.lineEdit_2.text()))

    dlg.show()
    app.exec()

if __name__ == "__main__":
    main()


    

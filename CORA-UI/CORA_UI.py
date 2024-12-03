from PyQt5 import QtWidgets, uic, QtGui
from PyQt5.QtGui import QColor, QTextCharFormat
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QListWidgetItem
import pandas as pd
import requests
import Feature_Extraction as FE
import pyqtgraph as pg
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QVBoxLayout



global Abstract, saved_citations
Abstract = ""
saved_citations = []

import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QVBoxLayout, QWidget

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QVBoxLayout, QWidget

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QVBoxLayout, QWidget


import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QVBoxLayout, QWidget


import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QVBoxLayout

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QVBoxLayout


def create_pie_chart(predictions):
    """
    Creates and displays a pie chart for a single prediction.
    Each class is represented as a slice of the pie, with percentages displayed.
    A legend (keymap) is added to show color labels for each class.
    """
    # Ensure predictions are valid
    if not predictions or not isinstance(predictions, list) or len(predictions) != 1:
        log_message("Error: Invalid predictions data. Expected a single prediction array.", color="red")
        return

    # Extract the single prediction (list of probabilities for 7 classes)
    prediction = predictions[0]

    # Define class labels (7 classes for the Cora dataset)
    class_labels = ["Case_Based", "Genetic_Algorithms", "Neural_Networks", "Probabilistic_Methods",
                    "Reinforcement_Learning", "Rule_Learning", "Theory"]

    # Ensure the prediction has 7 values (one for each class)
    if len(prediction) != len(class_labels):
        log_message("Error: Prediction does not contain exactly 7 values.", color="red")
        return

    # Show the graphWidget when generating the chart
    dlg.graphWidget.show()

    # Initialize a layout if the graphWidget has no layout
    if dlg.graphWidget.layout() is None:
        dlg.graphWidget.setLayout(QVBoxLayout())
    layout = dlg.graphWidget.layout()

    # Clear the existing layout in graphWidget
    for i in reversed(range(layout.count())):
        widget_to_remove = layout.itemAt(i).widget()
        layout.removeWidget(widget_to_remove)
        widget_to_remove.deleteLater()

    # Create the pie chart
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = plt.cm.Paired.colors  # Use a paired color map
    wedges, texts, autotexts = ax.pie(
        prediction,
        labels=None,  # Do not show labels on the pie chart
        autopct=lambda pct: f"{pct:.1f}%",  # Format percentages
        startangle=140,
        colors=colors  # Apply colors to slices
    )

    # Add a legend (keymap) to show class-color mapping
    ax.legend(
        wedges,  # Link legend to pie slices
        class_labels,  # Use class labels
        title="Classes",  # Legend title
        loc="center left",  # Position legend
        bbox_to_anchor=(1, 0, 0.5, 1),  # Position legend outside the pie chart
        fontsize=10
    )

    # Customize the pie chart
    ax.set_title("Predictions for Cora Classes", fontsize=14)
    for autotext in autotexts:
        autotext.set_fontsize(10)  # Set percentage font size
        autotext.set_color("white")  # Make percentages readable

    # Embed the matplotlib figure into the PyQt5 widget
    canvas = FigureCanvas(fig)
    canvas.draw()  # Explicitly draw the canvas to ensure it renders
    layout.addWidget(canvas)
    dlg.graphWidget.show()


def call_api(feature_vector, citations, api_url="http://127.0.0.1:5000/predict"):
    """
    Calls the prediction API with a given feature vector and citations.
    """
    payload = {
        "feature_vector": feature_vector,
        "connections": citations
    }
    try:
        response = requests.post(api_url, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.ConnectionError:
        return {"error": "Could not connect to API. Is the server running?"}
    except requests.Timeout:
        return {"error": "The API request timed out."}
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {e}"}

def map_citations_to_legitimate_titles_from_csv(citations, csv_file_path):
    """
    Maps the citation fake names to their legitimate titles using a CSV file.
    """
    try:
        nodes_with_legitimate_titles_df = pd.read_csv(csv_file_path)
    except Exception as e:
        log_message(f"Error reading CSV: {e}", color="red")
        return {}
    
    fake_name_to_legit_title = dict(zip(
        nodes_with_legitimate_titles_df.get("Fake Name", []),
        nodes_with_legitimate_titles_df.get("Top Paper Title", [])
    ))
    return {
        citation: fake_name_to_legit_title.get(citation, "Unknown Title")
        for citation in citations
    }

def log_message(message, color="white"):
    """Add a message to the log box."""
    fmt = QTextCharFormat()
    fmt.setForeground(QColor(color))
    dlg.log_box.setCurrentCharFormat(fmt)
    dlg.log_box.append(message)

def save_abstract():
    """Save the paper abstract."""
    global Abstract
    Abstract = dlg.textEdit.toPlainText().strip()
    if not Abstract:
        log_message("Error: Abstract cannot be empty.", color="red")
        return
    dlg.textEdit.setPlainText("")
    log_message("Paper Abstract saved: " + Abstract, color="blue")

def clear_abstract():
    """Clear the paper abstract."""
    global Abstract
    dlg.textEdit.setPlainText("")
    Abstract = ""
    log_message("Paper Abstract cleared.", color="blue")

def save_citations(legit_citations):
    """Save selected, checked, and entered citations, and uncheck all checkboxes."""
    global saved_citations
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
            # Uncheck the checkbox
            item.setCheckState(Qt.Unchecked)
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

def mark_searched_item(text):
    """Mark the searched item in the QListWidget."""
    for i in range(dlg.listWidget.count()):
        item = dlg.listWidget.item(i)
        if text.lower() in item.text().lower():
            item.setSelected(True)
        else:
            item.setSelected(False)

def clear_log_box():
    """
    Clears the content of the log box.
    """
    dlg.log_box.clear()
    log_message("Log box cleared.", color="blue")


def make_api_call():
    """Call the API with the saved citations and feature vector."""
    global Abstract, saved_citations
    if not Abstract:
        log_message("Error: Abstract is not saved. Cannot make API call.", color="red")
        return
    if not saved_citations:
        log_message("Error: No citations selected. Cannot make API call.", color="red")
        return
    
    feature_vector = FE.process_abstract(Abstract)
    # log_message(f"Making API call with feature vector: {feature_vector}", color="yellow")
    # log_message(f"Using citations: {saved_citations}", color="yellow")
    # log_message(str(sum(feature_vector)))
    response = call_api(feature_vector, saved_citations)
    if "predictions" in response:
        log_message("Predictions: " + str(response["predictions"]), color="green")
        create_pie_chart(response["predictions"])  # Pass predictions to the bar graph
    else:
        log_message("Error: " + response.get("error", "Unknown error occurred"), color="red")

def main():
    app = QtWidgets.QApplication([])
    global dlg
    dlg = uic.loadUi(r"C:\\Users\\mersh\\OneDrive\\Desktop\\CoraProject\\CORA-UI\\CORA.ui")
    dlg.graphWidget.hide()
    dlg.textEdit.setPlaceholderText("Enter Paper Abstract")
    dlg.lineEdit_2.setPlaceholderText("Search for a citation...")
    dlg.log_box.setReadOnly(True)
    dlg.listWidget.setSelectionMode(QtWidgets.QListWidget.MultiSelection)
    
    citations = [
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
    dlg.pushButton_2.clicked.connect(save_abstract)
    dlg.pushButton.clicked.connect(clear_abstract)
    dlg.pushButton_3.clicked.connect(lambda: save_citations(legit_citations))  # Save citations and uncheck boxes
    dlg.pushButton_4.clicked.connect(make_api_call)  # Make API call
    dlg.lineEdit_2.textChanged.connect(lambda: mark_searched_item(dlg.lineEdit_2.text()))
    dlg.pushButton_5.clicked.connect(clear_log_box)
    dlg.show()
    app.exec()

if __name__ == "__main__":
    main()

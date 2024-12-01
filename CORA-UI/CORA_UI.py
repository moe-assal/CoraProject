from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt 
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QListWidget, QListWidgetItem, QPushButton
import Feature_Extraction as FE

def log_message(message, color="white"):
    """Add a message to the log box."""
    dlg.log_box.setTextColor(QColor(color))
    dlg.log_box.append(message)  # Append the message to the log box

def save_paper(paper_name):
    # Check if the QTextEdit contains text
    if not dlg.textEdit.toPlainText().strip() == "":
        # Retrieve the text from QTextEdit
        paper_name = dlg.textEdit.toPlainText().strip()
        
        # Clear the QTextEdit
        dlg.textEdit.setPlainText("")
        
        # Log the saved paper name
        log_message("Paper Abstract: " + paper_name, color="blue")
    else:
        # Log an error message for empty input
        log_message("EMPTY STRING ENTERED", color="red")


def clear_paper(paper_abstract):
    # Reset the paper name variable
    paper_name = ""
    
    # Check if the QTextEdit contains text
    if not dlg.textEdit.toPlainText().strip() == "":
        # Clear the QTextEdit
        dlg.textEdit.setPlainText("")
        
        # Log that the paper name was cleared
        log_message("Paper Abstract cleared.", color="blue")
        log_message("Try Again: " + paper_name, color="blue")
        
        # Print the cleared paper name
        print(paper_name)
    else:
        # Log a message indicating that the QTextEdit is already empty
        log_message("Already Empty!", color="blue")


def save_citations(saved_citations):
    """Save selected, checked, and entered citations to a separate list."""
    for i in range(dlg.listWidget.count()):
        item = dlg.listWidget.item(i)
        if item.checkState() == Qt.Checked:  # Check if the checkbox is checked
            paper_id = item.text().split("_")[-1]  # Extract the ID at the end
            saved_citations.append(paper_id)

    selected_items = dlg.listWidget.selectedItems()
    for item in selected_items:
        paper_id = item.text().split("_")[-1]  # Extract the ID at the end
        if paper_id not in saved_citations:  # Avoid duplicates
            saved_citations.append(paper_id)

    entered_text = dlg.lineEdit_2.text().strip()
    if entered_text:
        item_found = False
        for i in range(dlg.listWidget.count()):
            item = dlg.listWidget.item(i)
            if entered_text.lower() == item.text().lower():  # Case-insensitive match
                paper_id = item.text().split("_")[-1]  # Extract the ID at the end
                if paper_id not in saved_citations:  # Avoid duplicates
                    saved_citations.append(paper_id)
                item_found = True
                break

        if not item_found:
            log_message(f"'{entered_text}' not found in the list!", color="red")

    if len(saved_citations) != 0:
        log_message("WARNING! ONCE SAVED, CITATIONS ARE VALID FOR 1-TIME USE", color="red")
        log_message("Citations saved successfully!", color="blue")
        log_message(str(saved_citations))

        for i in range(dlg.listWidget.count()):
            item = dlg.listWidget.item(i)
            item.setCheckState(Qt.Unchecked)
        dlg.listWidget.clearSelection()
        log_message("Selection Cleared!")
    else:
        log_message("INVALID Citations!", color="red")
        dlg.listWidget.clearSelection()

    dlg.lineEdit_2.clear()
    saved_citations.clear()
    log_message("Saved citations cleared after saving.", color="green")


def mark_searched_item(text):
    """Mark the searched item in the QListWidget."""
    for i in range(dlg.listWidget.count()):
        item = dlg.listWidget.item(i)
        if text.lower() in item.text().lower():
            item.setSelected(True)  # Select the matching item
        else:
            item.setSelected(False)  # Deselect non-matching items

def show_message(title, message):
    QMessageBox.information(None, title, message)

def main():
    app = QtWidgets.QApplication([])
    global dlg
    dlg = uic.loadUi(r"C:\\Users\\mersh\\OneDrive\\Desktop\\CoraProject\\CORA-UI\\CORA.ui")


    dlg.textEdit.setPlaceholderText("Enter Paper Abstract")
    abstract = dlg.textEdit.toPlainText()
    dlg.lineEdit_2.setPlaceholderText("Search for a citation...")
    dlg.log_box.setReadOnly(True)  # Make it read-only
    dlg.log_box.setPlaceholderText("Log messages will appear here...")
    dlg.listWidget.setSelectionMode(QListWidget.MultiSelection)  # Allow multiple selection
    dlg.lineEdit_2.textChanged.connect(lambda: mark_searched_item(dlg.lineEdit_2.text()))
    dlg.pushButton_2.clicked.connect(lambda: save_paper(abstract))
   

    dlg.pushButton.clicked.connect(lambda: clear_paper(abstract))

    abstract = dlg.textEdit.toPlainText()
    log_message(abstract, color="green")


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





    for citation in citations:
        item = QListWidgetItem(citation)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)  # Enable checkbox
        item.setCheckState(Qt.Unchecked)  # Set initial state to unchecked
        dlg.listWidget.addItem(item)

    saved_citations = []
    dlg.pushButton_3.clicked.connect(lambda: save_citations(saved_citations))
    
    feature_vector= FE.process_abstract(abstract)

    dlg.show()
    app.exec()

if __name__ == "__main__":
    main()

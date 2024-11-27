from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt 
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QListWidget, QListWidgetItem, QPushButton

def log_message(message, color="black"):
        """Add a message to the log box."""
        dlg.log_box.setTextColor(QColor(color))
        dlg.log_box.append(message)  # Append the message to the log box

def save_paper(paper_name):
    if not dlg.lineEdit.text() =="":
        paper_name =  dlg.lineEdit.text()
        dlg.lineEdit.setText("")
        log_message("Paper Name: " + paper_name, color="blue")
    else:
        log_message("EMPTY STRING ENTERED ", color="red")

def clear_paper(paper_name):
    paper_name=""
    if not dlg.lineEdit.text() =="":
        dlg.lineEdit.setText("")
        log_message("Paper name cleared: ", color="blue")
        log_message("Try Again" + paper_name, color="blue")
        print(paper_name)
    else:
        log_message("Already Empty! ",color="blue")
         
          
         

def save_citations(saved_citations):
    """Save selected, checked, and entered citations to a separate list."""

    # Save checked items
    for i in range(dlg.listWidget.count()):
        item = dlg.listWidget.item(i)
        if item.checkState() == Qt.Checked:  # Check if the checkbox is checked
            saved_citations.append(item.text())

    # Save selected items
    selected_items = dlg.listWidget.selectedItems()
    for item in selected_items:
        if item.text() not in saved_citations:  # Avoid duplicates
            saved_citations.append(item.text())

    # Check if text entered in lineEdit_2 matches any item
    entered_text = dlg.lineEdit_2.text().strip()
    if entered_text:
        item_found = False
        for i in range(dlg.listWidget.count()):
            item = dlg.listWidget.item(i)
            if entered_text.lower() == item.text().lower():  # Case-insensitive match
                if item.text() not in saved_citations:  # Avoid duplicates
                    saved_citations.append(item.text())
                item_found = True
                break

        if not item_found:
            log_message(f"'{entered_text}' not found in the list!", color="red")

    # Log warnings and success messages
    if len(saved_citations) != 0:
        log_message("WARNING! ONCE SAVED, CITATIONS ARE VALID FOR 1-TIME USE", color="red")
        log_message("Citations saved successfully!", color="blue")
        log_message(str(saved_citations))

        # Clear selection and reset checkboxes after saving
        for i in range(dlg.listWidget.count()):
            item = dlg.listWidget.item(i)
            item.setCheckState(Qt.Unchecked)
        dlg.listWidget.clearSelection()
        log_message("Selection Cleared!")
    else:
        log_message("INVALID Citations!", color="red")
        dlg.listWidget.clearSelection()

    # Clear the line edit and saved_citations
    dlg.lineEdit_2.clear()
    saved_citations.clear()
    log_message("Saved citations cleared after saving.", color="green")


def mark_searched_item(text):
        """Mark the searched item in the QListWidget."""
        for i in range(dlg.listWidget.count()):
            item = dlg.listWidget.item(i)
            # Check if the item matches the search text
            if text.lower() in item.text().lower():
                item.setSelected(True)  # Select the matching item
            else:
                item.setSelected(False)  # Deselect non-matching items


def show_message(title,message):
    QMessageBox.information(None,title,message)


app = QtWidgets.QApplication([])
dlg= uic.loadUi(r"C:\\Users\\mersh\\OneDrive\\Desktop\\CoraProject\\CORA-UI\\CORA.ui")

paper_name=""
dlg.lineEdit.setPlaceholderText("Enter Paper Name")
dlg.lineEdit_2.setPlaceholderText("Search for a citation...")
dlg.log_box.setReadOnly(True)  # Make it read-only
dlg.log_box.setPlaceholderText("Log messages will appear here...")
dlg.listWidget.setSelectionMode(QListWidget.MultiSelection)  # Allow multiple selection
dlg.lineEdit_2.textChanged.connect(lambda: mark_searched_item(dlg.lineEdit_2.text()))
dlg.pushButton_2.clicked.connect(lambda: save_paper(paper_name))
dlg.lineEdit.returnPressed.connect(lambda: save_paper(paper_name))
dlg.pushButton.clicked.connect(lambda: clear_paper(paper_name))
# Add citation links to the list
citations = [
            "1033 -> 1048", "1033 -> 1002", "1048 -> 1033",
            "1048 -> 1126", "1126 -> 1054", "1002 -> 1011",
            "1054 -> 1099", "1101 -> 1134", "1134 -> 1126",
            "2001 -> 2048", "2033 -> 2002", "2048 -> 2033", 
            "2048 -> 2126", "2126 -> 2054", "2202 -> 2111"
            ]

for citation in citations:
            item = QListWidgetItem(citation)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)  # Enable checkbox
            item.setCheckState(Qt.Unchecked)  # Set initial state to unchecked
            dlg.listWidget.addItem(item)
saved_citations=[]
dlg.pushButton_3.clicked.connect(lambda: save_citations(saved_citations))


dlg.show()
app.exec()
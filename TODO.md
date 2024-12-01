- please work in a new branch
- Step 1:
  - run the files class_imbalance.py, data_augmentation.py, node2vec.py
  - note that they are not well-tested, so some minor bugs may be present
  - save the final output (a large dictionary)
- Step 2:
  - use this large dictionary to train the models and create tables (number of options * number of gnns)
- Step 3:
  - based on the three tables generated, report back the best combination of options for every gnn
  - Train the 3 "best" gnns and save them in separate files
  - Show how they evolve during trained: (loss, epoch)-plot and (accuracy, epoch)-plot
- Step 4:
  - compare the 3 models using F1-score and show their Confusion matrices
- Step 5:
  - the best model across the 3 gnns will be moved to the UI
  - **we will keep the UI decoupled from ML-code**


- Note that files in analysis folder are important for writing the report.
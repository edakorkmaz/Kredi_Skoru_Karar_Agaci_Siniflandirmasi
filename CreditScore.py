import os
import pandas as pd
from sklearn.tree import export_graphviz
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score

# Specify the path to the Graphviz executables
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'  # Modify the path accordingly

# Veri setini yükleyin (örnek olarak bir dosya yolu verilmemiştir, lütfen uygun bir dosya yolu belirtin)
data = pd.read_csv('C:\\Users\\eda\\PycharmProjects\\pythonProject2\\dataset2.csv')  # Dosya yolunu doğru şekilde güncelleyin
print(data.head())
print(data.tail())
print(data.nunique())

# One-hot encoding uygula
data_encoded = pd.get_dummies(data, columns=['Gender', 'Education', 'Home Ownership'])
# Bağımsız değişkenleri ve hedef değişkeni seçin
X = data_encoded[['Age', 'Income', 'Gender_Male', 'Gender_Female', "Education_Doctorate", "Education_Bachelor's Degree" , 'Home Ownership_Rented' , 'Home Ownership_Owned']]
y = data_encoded['Credit Score']  # 'Target_Variable' gerçek hedef değişken adıyla değiştirilmelidir

# Veriyi eğitim ve test setlerine ayırın

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(criterion='entropy')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')


# Karar ağacını görselleştirin

dot_data = export_graphviz(model, out_file=None, feature_names=X.columns, class_names=model.classes_, filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph.render("decision_tree_c45")
graph.view()

# Eğitilmiş ağacın özetini yazdırın
tree_rules = export_text(model, feature_names=list(X.columns))
print(tree_rules)
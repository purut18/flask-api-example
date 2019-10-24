#!flask/bin/python
from flask import Flask, jsonify, make_response, request
from flask_httpauth import HTTPBasicAuth
import json
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

auth = HTTPBasicAuth()

app = Flask(__name__)

@auth.get_password
def get_password(username):
	if username == 'miguel':
		return 'python'
	return None

@auth.error_handler
def unauthorized():
	return make_response(jsonify({'error': 'Unauthorized'}))

iris_dataset = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'],random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)

tasks = [
	{
		'id': 1,
		'title': 'Buy Groceries',
		'description': 'Milk, Cheese, Pizza, Fruit',
		'done': False
	},
	{
		'id': 2,
		'title': 'Learn Python',
		'description': 'Need to find a good python tutorial',
		'done': True
	}
]

@app.route('/todo/api/tasks', methods=['GET'])
@auth.login_required
def get_tasks():
    return jsonify({'tasks': [5,2.9,1,0.2]})

@app.route('/todo/api/tasks/<int:task_id>', methods=['GET'])
def get_task(task_id):
	task = [task for task in tasks if task['id'] == task_id]
	if len(task) == 0:
		return make_response(jsonify({'error': 'Not Found'}), 404)
	return jsonify({'task': task[0]})

@app.route('/todo/api/tasks', methods=['POST'])
def add_task():
	if not request.json or not 'title' in request.json:
		abort(400)
	task = {
		'id': tasks[-1]['id'] + 1,
		'title': request.json['title'],
		'description': request.json.get('description', ""),
		'done': False
	}
	tasks.append(task)
	return jsonify({'task': task}), 201

@app.route('/todo/api/ml', methods=['POST'])
def return_prediction():
	if not request.json:
		abort(400)
	X_predict = np.array([json.loads(request.json['0'])])
	prediction = knn.predict(X_predict)
	return jsonify({"prediction": iris_dataset['target_names'][prediction].tolist()})


if __name__ == '__main__':
    app.run(debug=True)

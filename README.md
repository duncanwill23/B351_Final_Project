code to run in terminal after first pull starting:

#make enviroment
python -m venv env

get requirements:
pip install -r requirements.txt

#get dataset:
python main.py

#run project.py
python project.py

The analysis.py has functions used to help the project work. These include kmeans, cosine_distance, plot_clusters, plot_elbow_curve, recommend_movies, evaluate_clusters and evaluate_recommendations_manual.
the project.py has the main coding for the project, including the preprocessing.

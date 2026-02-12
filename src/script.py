import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def kmeans_clustering(all_vals, K, max_iter=100, tol=pow(10, -3)):
    n, m = all_vals.shape

    #Pick k random distinct data points as initial centroids
    init_idx = np.random.choice(n, K, replace=False)
    centroids = all_vals[init_idx].copy()

    assignments = np.zeros(n, dtype=int)
    all_sse = []

    for it in range(max_iter):
        #1) Assign points to nearest centroid
        #distances: shape (n,k)
        dists = np.linalg.norm(all_vals[:, None, :] - centroids[None, :, :], axis=2)
        assignments = np.argmin(dists, axis=1)

        #2) Compute new centroids
        new_centroids = np.zeros((K, m))
        for k in range(K):
            pts = all_vals[assignments == k]
            if len(pts) == 0:
                #Empty cluster? Reinitialize centroid randomly
                new_centroids[k] = all_vals[np.random.randint(0, n)]
            else:
                new_centroids[k] = pts.mean(axis=0)

        centroids = new_centroids

        #Compute sse: sum of squared distances to assigned centroid
        sse = 0.0
        for k in range(K):
            pts = all_vals[assignments == k]
            if len(pts) > 0:
                diffs = pts - centroids[k]
                sse += np.sum(np.linalg.norm(diffs, axis=1)**2)

        all_sse.append(sse)

        #Convergence check (only after first iteration)
        if it > 0:
            if np.absolute(all_sse[it] - all_sse[it-1]) / all_sse[it-1] <= tol:
                return assignments, centroids, all_sse, it+1

    #If we hit max_iter
    return assignments, centroids, all_sse, max_iter


def analyze_cluster_health(df, assignments):
    """Analyze each cluster to determine which represents healthy lifestyle"""
    cluster_stats = []
    
    for k in range(max(assignments) + 1):
        cluster_data = df[assignments == k]
        stats = {
            'cluster': k,
            'count': len(cluster_data),
            'avg_diabetes': cluster_data['diabetes'].mean(),
            'avg_exercises': cluster_data['exercises'].mean(),
            'avg_eats_well': cluster_data['eats well'].mean(),
            'avg_age': cluster_data['age'].mean(),
            'avg_weight': cluster_data['weight'].mean()
        }
        #Health score: higher exercise and eating well, lower diabetes
        stats['health_score'] = (stats['avg_exercises'] + stats['avg_eats_well']) - stats['avg_diabetes']
        cluster_stats.append(stats)
        #This currently essentially returns that the optimal strategy for health is being a toddler lol
    
    return cluster_stats


def get_user_input():
    """Get health statistics from user"""
    print("\n=== Health Lifestyle Assessment ===")
    print("Please enter your information:\n")
    
    while True:
        try:
            age = float(input("Age: "))
            if age < 0 or age > 120:
                print("Please enter a valid age.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")
    
    while True:
        try:
            weight = float(input("Weight (in pounds): "))
            if weight < 0:
                print("Please enter a valid weight.")
                continue
            break
        except ValueError:
            print("Please enter a valid number.")
    
    while True:
        try:
            exercises = int(input("Do you exercise regularly? (0 = No, 1 = Yes): "))
            if exercises not in [0, 1]:
                print("Please enter 0 or 1.")
                continue
            break
        except ValueError:
            print("Please enter 0 or 1.")
    
    while True:
        try:
            eats_well = int(input("Do you eat a healthy diet? (0 = No, 1 = Yes): "))
            if eats_well not in [0, 1]:
                print("Please enter 0 or 1.")
                continue
            break
        except ValueError:
            print("Please enter 0 or 1.")
    
    #Set diabetes to 0 for clustering purposes (we'll use cluster stats instead)
    diabetes = 0
    
    return np.array([[age, weight, diabetes, exercises, eats_well]])


def classify_user(user_data, centroids):
    """Classify user into nearest cluster"""
    dists = np.linalg.norm(centroids - user_data, axis=1)
    return np.argmin(dists)


def main():
    #Load the data
    df = pd.read_csv('dummydata.csv')
    df.columns = ['age', 'weight', 'diabetes', 'exercises', 'eats well']
    x = df[['age', 'weight', 'diabetes', 'exercises', 'eats well']].values

    #Run k-means clustering
    k = 4
    assignments, centroids, all_sse, iters = kmeans_clustering(x, k)
    assignments = np.array(assignments).reshape(-1)

    #Analyze clusters to determine health characteristics
    cluster_stats = analyze_cluster_health(df, assignments)
    
    print("\nCluster Analysis from Training Data:")
    print("=" * 70)
    for stats in sorted(cluster_stats, key=lambda x: x['health_score'], reverse=True):
        print(f"\nCluster {stats['cluster']} (n={stats['count']}):")
        print(f"  Average age: {stats['avg_age']:.1f}")
        print(f"  Average weight: {stats['avg_weight']:.1f} lbs")
        print(f"  Diabetes rate: {stats['avg_diabetes']*100:.1f}%")
        print(f"  Exercise rate: {stats['avg_exercises']*100:.1f}%")
        print(f"  Healthy eating rate: {stats['avg_eats_well']*100:.1f}%")
        print(f"  Health score: {stats['health_score']:.2f}")
    
    #Identify healthiest cluster
    healthiest_cluster = max(cluster_stats, key=lambda x: x['health_score'])['cluster']
    
    #Get user input
    user_data = get_user_input()
    
    #Classify user
    user_cluster = classify_user(user_data, centroids)
    
    #Generate assessment
    print("\n" + "=" * 70)
    print("ASSESSMENT RESULTS")
    print("=" * 70)
    print(f"\nYou have been classified into Cluster {user_cluster}")
    
    user_cluster_stats = cluster_stats[user_cluster]
    print(f"\nPeople in this cluster have:")
    print(f"  - {user_cluster_stats['avg_diabetes']*100:.1f}% diabetes rate")
    print(f"  - {user_cluster_stats['avg_exercises']*100:.1f}% exercise regularly")
    print(f"  - {user_cluster_stats['avg_eats_well']*100:.1f}% eat a healthy diet")
    print(f"  - Average age: {user_cluster_stats['avg_age']:.1f} years")
    print(f"  - Average weight: {user_cluster_stats['avg_weight']:.1f} lbs")
    
    #Calculate relative diabetes risk
    overall_diabetes_rate = df['diabetes'].mean()
    relative_risk = user_cluster_stats['avg_diabetes'] / overall_diabetes_rate if overall_diabetes_rate > 0 else 1.0
    
    if relative_risk > 1.2:
        risk_level = "HIGHER"
        risk_color = "⚠"
    elif relative_risk < 0.8:
        risk_level = "LOWER"
        risk_color = "✓"
    else:
        risk_level = "AVERAGE"
        risk_color = "•"
    
    print(f"\n{risk_color} Based on this cluster, your diabetes risk is {risk_level} than average")
    print(f"  (Cluster rate: {user_cluster_stats['avg_diabetes']*100:.1f}% vs Overall rate: {overall_diabetes_rate*100:.1f}%)")
    
    if user_cluster == healthiest_cluster:
        print("\n✓ Your lifestyle appears to be HEALTHY!")
        print("  You're in the cluster with the best health indicators.")
        print("  Keep up the good work!")
    else:
        print("\n⚠ Your lifestyle could be IMPROVED.")
        print(f"  The healthiest cluster (Cluster {healthiest_cluster}) has:")
        healthiest_stats = cluster_stats[healthiest_cluster]
        
        improvements = []
        if user_data[0][3] == 0 and healthiest_stats['avg_exercises'] > 0.5:
            improvements.append("Start exercising regularly")
        if user_data[0][4] == 0 and healthiest_stats['avg_eats_well'] > 0.5:
            improvements.append("Improve your diet")
        
        #Show potential diabetes risk reduction
        if user_cluster_stats['avg_diabetes'] > healthiest_stats['avg_diabetes']:
            risk_reduction = ((user_cluster_stats['avg_diabetes'] - healthiest_stats['avg_diabetes']) / 
                            user_cluster_stats['avg_diabetes'] * 100)
            improvements.append(f"Potentially reduce diabetes risk by {risk_reduction:.0f}% with lifestyle changes")
        
        if improvements:
            print("\n  Suggestions for improvement:")
            for imp in improvements:
                print(f"    • {imp}")
    
    #Visualize with t-SNE
    print("\nGenerating visualization...")
    tsne = TSNE(n_components=2, perplexity=30, init='random', random_state=0)
    embedding = tsne.fit_transform(x)
    
    #Also transform user data (approximate by adding to dataset and transforming)
    x_with_user = np.vstack([x, user_data])
    tsne_with_user = TSNE(n_components=2, perplexity=30, init='random', random_state=0)
    embedding_with_user = tsne_with_user.fit_transform(x_with_user)
    
    plt.figure(figsize=(10, 6))
    
    #Plot training data
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=assignments,
        s=30,
        alpha=0.6,
        cmap='viridis',
        label='Training data'
    )
    
    #Plot user point
    plt.scatter(
        embedding_with_user[-1, 0],
        embedding_with_user[-1, 1],
        c='red',
        s=200,
        marker='*',
        edgecolors='black',
        linewidths=2,
        label='You',
        zorder=5
    )
    
    plt.colorbar(scatter, label='Cluster')
    plt.title('t-SNE Visualization: Your Health Profile vs Training Data')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/health_assessment_plot.png', dpi=150, bbox_inches='tight')
    print("Visualization saved!")
    plt.show()


if __name__ == "__main__":
    main()
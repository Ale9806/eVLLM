import pandas as pd
import matplotlib.pyplot as plt

def plot_selected_columns(csv_file, columns_to_plot):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Extract the column names and convert them to a list
    column_names = df.columns.tolist()

    # Remove the first column which is assumed to be the index
    column_names.pop(0)

    # Extract the column names to plot based on user input
    columns_to_plot_names = [column_names[i] for i in columns_to_plot]

    # Select only the desired columns for plotting
    df_to_plot = df.iloc[:, columns_to_plot]

    # Set the 'model' column as the index for better labeling
    df_to_plot.set_index('model', inplace=True)

    # Plotting
    df_to_plot.plot(kind='bar', figsize=(10, 6))
    plt.title('Bar Plot of Selected Columns')
    plt.xlabel('Model')
    plt.ylabel('Values')
    plt.legend(columns_to_plot_names)
    plt.show()

# Example usage:
# Provide the path to your CSV file and the indices of the columns you want to plot
csv_file_path = 'outputs/tables/eval.csv'
columns_to_plot_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # Adjust indices as needed
plot_selected_columns(csv_file_path, columns_to_plot_indices)
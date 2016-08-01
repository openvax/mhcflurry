from mhcflurry.dataset import Dataset
from mhcflurry.peptide_encoding import indices_to_hotshot_encoding

file_to_explore="/root/.local/share/mhcflurry/2/class1_data/combined_human_class1_dataset.csv"
dataset = Dataset.from_csv(
        filename=file_to_explore,
        sep=",",
        peptide_column_name="peptide")
df = dataset.to_dataframe()
df_kmers = dataset.kmer_index_encoding()
training_hotshot = indices_to_hotshot_encoding(df_kmers[0])

from keras.models import Sequential
from keras.layers import Dense, Activation
model = Sequential()
model.add(Dense(input_dim=9*21, output_dim=1))

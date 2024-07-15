# %%
from AstridEmbed import *

# %%
query_type = "substring"
dataset = "imdb_movie_actors"


# def get_model(query_type, dataset):
random_seed = 1234
misc_utils.initialize_random_seeds(random_seed)

# Set the configs
embedding_learner_configs, frequency_configs, selectivity_learner_configs = (
    setup_configs(query_type, dataset)
)

embedding_model_file_name = selectivity_learner_configs.embedding_model_file_name
selectivity_model_file_name = (
    selectivity_learner_configs.selectivity_model_file_name
)

string_helper = misc_utils.setup_vocabulary(frequency_configs.string_list_file_name)
embedding_model = load_embedding_model(embedding_model_file_name, string_helper).to(
    "cuda"
)
selectivity_model = load_selectivity_estimation_model(
    selectivity_model_file_name, string_helper
).to("cuda")
# to load max min
df = pd.read_csv(frequency_configs.selectivity_file_name)
df["string"] = df["string"].astype(str)
df = compute_normalized_selectivities(df)
#     return embedding_model, selectivity_model, string_helper


# embed_model, sel_model, str_helper = get_model(query_type, dataset)

# %%
df

# %%
#Load the input file and split into 50-50 train, test split
df = pd.read_csv(frequency_configs.selectivity_file_name)
df["string"] = df["string"].astype(str)
df = compute_normalized_selectivities(df)
train_indices, test_indices = train_test_split(df.index, random_state=random_seed, test_size=0.5)
train_df, test_df = df.iloc[train_indices], df.iloc[test_indices]

# %%
test_df

# %%
#Get the predictions from the learned model and compute basic summary statistics
normalized_predictions, denormalized_predictions = get_selectivity_for_strings(
    test_df["string"].values, embedding_model, selectivity_model, string_helper)
actual = torch.tensor(test_df["normalized_selectivities"].values)
test_q_error = misc_utils.compute_qerrors(normalized_predictions, actual,
    selectivity_learner_configs.min_val, selectivity_learner_configs.max_val)
print("Test data: Mean q-error loss ", np.mean(test_q_error))
print("Test data: Summary stats of Loss: Percentile: [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99] ", [np.quantile(test_q_error, q) for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]])

# %%
test_df["pred"] = normalized_predictions.cpu().numpy()
test_df["dpred"] = denormalized_predictions.cpu().numpy()
test_df.head(300)

# %%
test_df[["normalized_selectivities","pred"]].head()

# %%
q_error = []
for _, row in test_df[["dpred","selectivity"]].iterrows():
    pred = max(row.dpred, 0.00000000001)
    target = max(row.selectivity, 0.00000000001)
    if (pred > target):
        q_error.append(pred / target)
    else:
        q_error.append(target / pred)
np.median(q_error), np.mean(q_error), [np.quantile(q_error, q) for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]]

# %%
pd.set_option('display.max_rows', None)
test_df.sort_values(by="normalized_selectivities", ascending=False).head(10)

# %%


# %%
selectivity_learner_configs

# %%
misc_utils.unnormalize_torch

# %%
normalized_predictions, denormalized_predictions = get_selectivity_for_strings(
    ["The", "The Lord", "a"], embedding_model, selectivity_model, string_helper
)
#misc_utils.unnormalize_torch(normalized_predictions, 0.0, 15.0)
for pred in denormalized_predictions:
    print(pred)

# %%
test_df["string"].values

# %%
default_preds_path = "benchmark/preds_true.txt"
output_preds_path = "benchmark/preds_ASTRID.txt"
columns_to_replace = [
    # "t.title",
    "n.name",
]

pred_lines = [pred for pred in open(default_preds_path).read().strip().splitlines()]

print(len(pred_lines), pred_lines[1])

# %%
def detect_like_type(line):
    w: str = line.split(",")[0].split(" ")[2]
    if w.count("%") == 2:
        return "substring"
    elif w[1] == "%":
        return "suffix"
    elif w[-2] == "%":
        return "prefix"
    else:
        raise ValueError(f"Undefined like type: {w} {line}")

# %%
def get_astrid_word(line):
    w: str = line.split(",")[0].split(" ")[2]
    return w.replace("'", "").replace("%", "")

# %%
get_astrid_word("n.name ~~ 'Christi%',0.005000")

# %%
query_types = ["prefix", "suffix", "substring"]
datasets = ["imdb_movie_actors"]
n_rows = [4167486]
column_names = ["n.name",]

for query_type in query_types:
    for dataset, column_name, n_row in zip(datasets, column_names, n_rows):
        print(dataset, query_type, column_name)
        embed_model, sel_model, str_helper = get_model(query_type, dataset)
        for i, line in enumerate(pred_lines):
            if line.startswith("-- query"):
                continue
            if line.startswith(column_name) and " ~~ " in line:
                if detect_like_type(line) == query_type:
                    norm_pred, denorm_pred = get_selectivity_for_strings(
                        [get_astrid_word(line)], embed_model, sel_model, str_helper
                    )
                    parts = line.split(",")
                    sel = denorm_pred.item() / n_row
                    print(i, line, f"{sel:.6f}")
                    pred_lines[i] = f"{parts[0]},{sel:.6f}"

# write the updated output
f = open(output_preds_path, "w+")
f.writelines(f"{s}\n" for s in pred_lines)
f.close

# %%
test_df["selectivity"].hist()

# %%
test_df.head()

# %%
(test_df["selectivity"] < 10).sum()

# %%
(test_df["selectivity"] < 10).sum() /len(test_df["selectivity"])

# %%
list_ours = []
column_name = "n.name"
pred_lines = [pred for pred in open(default_preds_path).read().strip().splitlines()]

for i, line in enumerate(pred_lines):
    if line.startswith("-- query"):
        continue
    if line.startswith(column_name) and " ~~ " in line:
        parts = line.split("'")
        list_ours.append(int(parts[2][1:]))

# %%
cdf = pd.DataFrame({"c":list_ours})
cdf.hist()

# %%
(cdf["c"] < 10).sum() /len(cdf["c"])

# %%




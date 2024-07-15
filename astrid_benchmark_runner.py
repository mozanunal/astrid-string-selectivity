# %%
from AstridEmbed import *
import copy
import sys
from tqdm import tqdm

sys.path.append("/home/kiymet/postgres-like-selectivity-benchmark/")
from pg_like_bench.pg_types import LikeExpr

# %%
query_type = "substring"
dataset = "imdb_movie_actors"
like_exprs_path = "~/postgres-like-selectivity-benchmark/benchmarks/imdb_10k/exprs.csv"



# %%
def get_model(query_type, dataset):
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
    return embedding_model, selectivity_model, string_helper


embed_model, sel_model, str_helper = get_model(query_type, dataset)

# %%
selectivity_learner_configs

# %%
normalized_predictions, denormalized_predictions = get_selectivity_for_strings(
    ["The", "b", "a"], embed_model, sel_model, str_helper
)
#misc_utils.unnormalize_torch(normalized_predictions, 0.0, 15.0)
for pred in denormalized_predictions:
    print(pred)



# %%
query_types = ["prefix", "suffix", "substring"]
datasets = ["imdb_movie_actors",]# "imdb_movie_titles"]
n_rows = [4167486]#, 2527952]
column_names = ["n.name"]#, "t.title"]
like_list = LikeExpr.parse_file(like_exprs_path)

for query_type in query_types:
    for dataset, column_name, n_row in zip(datasets, column_names, n_rows):
        print(dataset, query_type, column_name)
        embed_model, sel_model, str_helper = get_model(query_type, dataset)
        for expr in tqdm(like_list):
            expr: LikeExpr
            norm_pred, denorm_pred = get_selectivity_for_strings(
                [expr.like_expr.replace('%', '')], embed_model, sel_model, str_helper
            )
            expr.astrid = int(denorm_pred.item())
LikeExpr.dump_to_file(like_exprs_path, like_list)

# %%



# 

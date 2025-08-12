import os
from datasets import load_dataset
import pyarrow as pa
import pyarrow.parquet as pq


def safe_sample_and_save(dataset_path, output_path, sample_size=1000, seed=42):
    try:
        from fastparquet import ParquetFile
        dataset = ParquetFile(dataset_path)
        df = dataset.to_pandas()

        # df采样
        sampled = df.sample(n=sample_size, random_state=seed)

        sampled_table = pa.Table.from_pandas(sampled)

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 保存为Parquet
        pq.write_table(sampled_table, output_path, compression='snappy')
        print(f"🎯 采样完成！输出文件: {os.path.abspath(output_path)}")
        print(f"📊 文件大小: {os.path.getsize(output_path) / (1024 ** 2):.2f} MB")
        print(f"📋 实际采样数量: {sampled_table.num_rows}")  # 验证样本量

    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        raise


# 使用示例
safe_sample_and_save(
    dataset_path="./rlpr_train.parquet",
    output_path="./dist_entropy_rlpr_1k.parquet",
    sample_size=1000,
    seed=42
)
"""De-duplicate results_diary.csv — keep last entry per experiment_id."""
import pandas as pd

diary = pd.read_csv('diary/results_diary.csv')
print(f"Before: {len(diary)} rows, {diary['experiment_id'].nunique()} unique")

# Keep last occurrence of each experiment_id
deduped = diary.drop_duplicates(subset='experiment_id', keep='last')
print(f"After:  {len(deduped)} rows, {deduped['experiment_id'].nunique()} unique")
print(f"Removed: {len(diary) - len(deduped)} duplicate rows")

deduped.to_csv('diary/results_diary.csv', index=False)
print("Saved.")

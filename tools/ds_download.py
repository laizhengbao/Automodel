import os
import zipfile
import tarfile
from huggingface_hub import HfApi, hf_hub_download

def download_and_extract_all_archives(
	repo_id: str, 
	extract_dir: str, 
	repo_type: str = "dataset"
) -> str:
	os.makedirs(extract_dir, exist_ok=True)
	
	api = HfApi()
	
	print(f"Scanning files in {repo_id}...")

	all_files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
	
	archives = [f for f in all_files if f.endswith(('.zip', '.tar.gz', '.tgz'))]
	
	if not archives:
		print( "No archives found, please check your input", file=os.sys.stderr )
		return extract_dir
		
	print(f"Found {len(archives)} archive(s)")
	
	for filename in archives:
		safe_name = filename.replace("/", "_")
		success_flag_path = os.path.join(extract_dir, f".{safe_name}_extracted")
		
		if os.path.exists(success_flag_path):
			print(f"[{filename}] extracted. skipping...")
			continue
			
		print(f"\n--- Processing {filename} ---")
		try:
			print("Downloading...")
			file_path = hf_hub_download(
				repo_id=repo_id,
				filename=filename,
				repo_type=repo_type
			)
			
			print(f"Extracting to {extract_dir} ...")
			if filename.endswith(".zip"):
				with zipfile.ZipFile(file_path, 'r') as zip_ref:
					zip_ref.extractall(extract_dir)
			elif filename.endswith((".tar.gz", ".tgz")):
				with tarfile.open(file_path, 'r:gz') as tar_ref:
					tar_ref.extractall(extract_dir)
					
			with open(success_flag_path, 'w') as f:
				f.write("done")
			print("Finished!")
			
		except Exception as e:
			print(f"Get error {e} while processing {filename}")
			
	print("\nAll the archives are done!")
	return extract_dir


if __name__ == "__main__":

	ds, save = None, None
	tp = "dataset"

	al = len( os.sys.argv )
	if al > 1:
		ds = os.sys.argv[1]
	if al > 2:
		save = os.sys.argv[2]
	if al > 3:
		tp = os.sys.argv[3]

	bd = download_and_extract_all_archives (
		ds,
		save,
		tp,
	)

	print( f"Base dir: {bd}" )

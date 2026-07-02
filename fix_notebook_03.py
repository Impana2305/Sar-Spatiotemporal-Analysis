import json

notebook_path = "03_SpatioTemporal_Analysis_Outputs.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "self.pre_proj = nn.Linear(spatial_dim, embed_dim)" in source:
            new_source = source.replace("self.pre_proj", "self.feature_proj").replace("self.post_proj", "self.feature_unproj")
            
            # Split new source by newlines and append \n except for the last line
            new_source_lines = [line + "\n" for line in new_source.split("\n")]
            new_source_lines[-1] = new_source_lines[-1].rstrip("\n")
            cell["source"] = new_source_lines
            break

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook 03 patched successfully!")

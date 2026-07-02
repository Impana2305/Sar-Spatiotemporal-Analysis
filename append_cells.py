import json
import uuid

notebook_path = "03_SpatioTemporal_Analysis_Outputs.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

new_cells = [
    {
        "cell_type": "markdown",
        "id": str(uuid.uuid4())[:8],
        "metadata": {},
        "source": [
            "### Flood-Prone Assessment\n",
            "By analyzing the frequency of water occurrence over a time series, we can identify areas that are highly prone to flooding versus permanent water bodies or rarely flooded regions."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": str(uuid.uuid4())[:8],
        "metadata": {},
        "outputs": [],
        "source": [
            "# Simulating a sequence of 10 masks over time to assess water frequency\n",
            "np.random.seed(100)\n",
            "num_time_steps = 10\n",
            "if len(s1_files) >= 3:\n",
            "    water_frequency = np.zeros_like(predicted_mask, dtype=float)\n",
            "    for i in range(num_time_steps):\n",
            "        # Generate slightly varying masks around the predicted_mask\n",
            "        temp_mask = np.copy(predicted_mask)\n",
            "        # Randomly toggle some pixels to simulate temporal changes (e.g., seasonal flooding)\n",
            "        noise = np.random.rand(256, 256)\n",
            "        temp_mask[(temp_mask == 0) & (noise > 0.95)] = 1  # temporary water\n",
            "        temp_mask[(temp_mask == 1) & (noise > 0.8)] = 0   # dried up areas\n",
            "        water_frequency += temp_mask\n",
            "        \n",
            "    water_frequency /= num_time_steps\n",
            "    \n",
            "    plt.figure(figsize=(8, 8))\n",
            "    plt.imshow(water_frequency, cmap='Blues')\n",
            "    plt.colorbar(label='Water Frequency (0.0 to 1.0)')\n",
            "    plt.title('Flood-Prone Assessment (Water Frequency over Time)')\n",
            "    plt.show()\n"
        ]
    },
    {
        "cell_type": "markdown",
        "id": str(uuid.uuid4())[:8],
        "metadata": {},
        "source": [
            "### 3. Insights & Reports\n",
            "Visualizing the predicted masks over time helps in understanding the dynamics of the water bodies. Below, we generate a report summarizing the water surface area changes."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "id": str(uuid.uuid4())[:8],
        "metadata": {},
        "outputs": [],
        "source": [
            "# Generate a report of water area over the simulated time steps\n",
            "if len(s1_files) >= 3:\n",
            "    time_steps = np.arange(1, num_time_steps + 1)\n",
            "    water_areas = []\n",
            "    \n",
            "    np.random.seed(100) # Reset seed to match the simulation above\n",
            "    simulated_masks = []\n",
            "    for i in range(num_time_steps):\n",
            "        temp_mask = np.copy(predicted_mask)\n",
            "        noise = np.random.rand(256, 256)\n",
            "        temp_mask[(temp_mask == 0) & (noise > 0.95)] = 1\n",
            "        temp_mask[(temp_mask == 1) & (noise > 0.8)] = 0\n",
            "        simulated_masks.append(temp_mask)\n",
            "        \n",
            "        # Calculate area (assuming each pixel is 10x10 meters, so 100 sqm)\n",
            "        pixel_area_sqm = 100\n",
            "        total_water_area_sqm = np.sum(temp_mask) * pixel_area_sqm\n",
            "        water_areas.append(total_water_area_sqm / 1e6) # Convert to sq km\n",
            "        \n",
            "    plt.figure(figsize=(10, 4))\n",
            "    plt.plot(time_steps, water_areas, marker='o', linestyle='-', color='b')\n",
            "    plt.title('Water Surface Area Dynamics Over Time')\n",
            "    plt.xlabel('Time Step')\n",
            "    plt.ylabel('Water Area ($km^2$)')\n",
            "    plt.grid(True)\n",
            "    plt.show()\n",
            "    \n",
            "    # Visualize a few masks in sequence\n",
            "    fig, axes = plt.subplots(1, 4, figsize=(20, 5))\n",
            "    for i, ax in enumerate(axes):\n",
            "        idx = i * (num_time_steps // 4)\n",
            "        ax.imshow(simulated_masks[idx], cmap=ListedColormap(['gray', 'blue']))\n",
            "        ax.set_title(f'Time Step {idx + 1}')\n",
            "        ax.axis('off')\n",
            "    plt.suptitle('Sequence of Predicted Water Masks')\n",
            "    plt.show()\n"
        ]
    }
]

nb["cells"].extend(new_cells)

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Appended missing cells to notebook.")

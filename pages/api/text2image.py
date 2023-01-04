import torch
from tqdm.auto import tqdm
print('importing models...')
from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud

"""### Models"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('creating base model...')
base_name = 'base40M-textvec'
base_model = model_from_config(MODEL_CONFIGS[base_name], device)
base_model.eval()
base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

print('creating upsample model...')
upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
upsampler_model.eval()
upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

print('downloading base checkpoint...')
base_model.load_state_dict(load_checkpoint(base_name, device))

print('downloading upsampler checkpoint...')
upsampler_model.load_state_dict(load_checkpoint('upsample', device))

sampler = PointCloudSampler(
    device=device,
    models=[base_model, upsampler_model],
    diffusions=[base_diffusion, upsampler_diffusion],
    num_points=[1024, 4096 - 1024],
    aux_channels=['R', 'G', 'B'],
    guidance_scale=[3.0, 0.0],
    model_kwargs_key_filter=('texts', ''), # Do not condition the upsampler at all
)

import sys

# Read the first command-line argument
prompt = sys.argv[1]

realPrompt = prompt
realPrompt = realPrompt.replace(' ', '_')
print(realPrompt)
# Produce a sample from the model.
samples = None
for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt]))):
    samples = x
    break
print('sampled')
# Plot the sample.

pc = sampler.output_to_point_clouds(samples)[0]

fig = plot_point_cloud(pc, grid_size=3, fixed_bounds=((-0.75, -0.75, -0.75),(0.75, 0.75, 0.75)))

import plotly.graph_objects as go

fig_plotly = go.Figure(
        data=[
            go.Scatter3d(
                x=pc.coords[:,0], y=pc.coords[:,1], z=pc.coords[:,2], 
                mode='markers',
                marker=dict(
                  size=2,
                  color=['rgb({},{},{})'.format(r,g,b) for r,g,b in zip(pc.channels["R"], pc.channels["G"], pc.channels["B"])],
              )
            )
        ],
        layout=dict(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False)
            )
        ),
    )

fig_plotly.show(renderer="colab")

from point_e.util.pc_to_mesh import marching_cubes_mesh

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('creating SDF model...')
name = 'sdf'
model = model_from_config(MODEL_CONFIGS[name], device)
model.eval()

print('loading SDF model...')
model.load_state_dict(load_checkpoint(name, device))

import skimage.measure as measure

# Produce a mesh (with vertex colors)
mesh = marching_cubes_mesh(
    pc=pc,
    model=model,
    batch_size=4096,
    grid_size=32, # increase to 128 for resolution used in evals
    progress=True,
)

import cockroachdb
import psycopg2
import io

# Connect to the cockroach database.

conn = psycopg2.connect(
    dbname='fuse-app-4550.files',
    user='akshay',
    password='fvjwTAErIHY3bVVSeWrLjA',
    host= 'fuse-app-4550.6wr.cockroachlabs.cloud',
    port=26257,
    sslmode = 'require',
)

cursor = conn.cursor()

# Use the cursor to execute a CREATE TABLE statement
cursor.execute(f"""
    CREATE TABLE {realPrompt}files (
        id SERIAL PRIMARY KEY,
        name TEXT NOT NULL,
        data BYTEA NOT NULL
    )
""")

# Commit the transaction
conn.commit()

# Create a cursor
cursor = conn.cursor()

binary_data = io.BytesIO()
mesh.write_ply(binary_data)
binary_data = binary_data.getvalue()

# Use the cursor to execute an INSERT statement to store the binary object in the database
cursor.execute(f"INSERT INTO {realPrompt}files (name, data) VALUES (%s, %s)", (realPrompt+'.ply', binary_data))
print('inserted')

# Commit the transaction
conn.commit()

import trimesh

# Create a cursor
cursor = conn.cursor()

# Use the cursor to execute a SELECT statement to retrieve the desired row
cursor.execute(f"SELECT data FROM {realPrompt}files WHERE name = '{realPrompt}.ply'")

# Retrieve the binary data from the data column
binary_data = cursor.fetchone()[0]


# Convert the binary data to a Trimesh mesh object
ply_mesh = trimesh.load_mesh(io.BytesIO(binary_data), file_type= 'ply')

# Write the STL file
binary_data = io.BytesIO()
ply_mesh.export(binary_data, file_type='stl')
binary_data = binary_data.getvalue()

cursor.execute(f"INSERT INTO {realPrompt}files (name, data) VALUES (%s, %s)", (realPrompt+'.stl', binary_data))

conn.commit()

import os
""" from pathlib import Path
downloads_path = str(Path.home() / "Downloads") """

downloads_folder = os.path.join(os.path.expanduser('~'), 'Downloads')

file_path = os.path.join(downloads_folder, realPrompt + '.stl')

with open(file_path, 'wb') as f:
    f.write(binary_data)

import numpy as np
import stl
import tempfile

cursor = conn.cursor()
cursor.execute(f"SELECT data FROM {realPrompt}files WHERE name = '{realPrompt}.stl'")
stl_file = cursor.fetchone()[0]

# Convert the memory view to a bytes object
stl_bytes = bytes(stl_file)
print(stl_bytes)

# Create a temporary file and write the stl_bytes object to it
with tempfile.NamedTemporaryFile(mode='w+b', suffix='.stl', delete=False) as fp:
    fp.write(stl_bytes)
    stl_file = fp.name

mesh = stl.mesh.Mesh.from_file(stl_file)

vertices = mesh.vectors

print((vertices))
image = ''
# Write the Gcode header
image += ('G28\n')  # Home all axes
image +=('G1 F6000\n')  # Set feed rate
image +=('G92 E0\n')  # Set extruder position to 0
image +=('G1 Z0.2\n')  # Move the nozzle to a safe height

# Iterate over the vertices of the mesh and write Gcode commands
for vertex in vertices:
  for vert in vertex:
    x, y, z = vert[0]+100,vert[1]+100,vert[2]  # Get the coordinates of the vertex
    image +=(f'G1 X{x} Y{y} E{z};\n')  # Move the nozzle to the vertex position
    image +=('G1 E3\n')  # Extrude 1mm of material
    image +=('G4 P100\n')  # Wait 100ms

image = bytes(image, encoding='utf-8')

cursor.execute(f"INSERT INTO {realPrompt}files (name, data) VALUES ('{realPrompt}.gcode', {image})")
print('gcode file inserted')

# Commit the transaction
conn.commit()

# Close the cursor and connection
cursor.close()
conn.close()

# Delete the temporary file
import os
os.unlink(stl_file)
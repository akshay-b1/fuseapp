from flask import Flask
from flask import request
from flask import render_template
import smtplib
from flask_mail import Mail, Message


app = Flask(__name__, template_folder='templates', static_folder='static')
mail = Mail(app)

app.config['MAIL_SERVER']='smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'fuseapp3d@gmail.com'
app.config['MAIL_PASSWORD'] = 'pyplttezgvykvztr'
app.config['MAIL_USE_SSL'] = True
app.config['MAIL_USE_TLS'] = False
mail = Mail(app)

@app.route('/contact', methods=['POST','GET'])
def send_message():
    if request.method == "POST":
        print("inside")
        msg = Message(request.form['subject'] + " | Customer Email: " + request.form['email'], sender = 'fuseapp3d@gmail.com', recipients = [request.form['email'], 'fuseapp3d@gmail.com'])
        msg.body = request.form['message']
        print("here")
        print(msg)
        mail.send(msg)
        return render_template('contact.html')
    else:
        return render_template('contact.html')

@app.route('/FAQ')
def FAQ():
    return render_template('FAQ.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/market')
def market():
    return render_template('market.html')



@app.route('/prototyping')
def prototyping():
    return render_template('prototyping.html')

@app.route('/pdesign')
def pdesign():
    return render_template('pdesign.html')
    
from werkzeug.utils import secure_filename
@app.route('/picTo3D', methods=['POST','GET'])
def image2image():
    files = ""
    name = ""
    if request.method == 'POST':
      files = request.files['image']
      name = secure_filename(files.filename)
      files.save(secure_filename(files.filename))
    if files:
        fileName = file_from2(files, name)
    else:
        fileName = ""
    return render_template('picTo3D.html', fileName=fileName)

def file_from2(files, name):
    try:
        
        if name.count("/") > 0:
            end = name.split("/")[-1]
        else:
            end = name
        ind = end.index(".")
        word = "img" + end[:ind]
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
        print("connected")
        query = f"SELECT * FROM images"
        print('got image')

    # Execute the insert query
        cursor = conn.cursor()
        cursor.execute(query)
        
        # Fetch the results
        results = cursor.fetchall()
        entered = False
        temp = []
        for row in results:
            name = row
            temp.append(name)
            entered = True
        if not entered:
            query = f"INSERT INTO images (image) VALUES ('{word + str(0)}')"
            word = word+str(0)
            # Execute the search query
            cursor = conn.cursor()
            cursor.execute(query)
        else:
            enteredAgain = False
            for i in temp:
                war = i[1]
                if war[:len(i[1])-1] == word:
                    enteredAgain = True
                    name = i[1]
                    print(name)
                    break
            if enteredAgain:
                word = word+str(int(name[-1])+1)
                query = f"INSERT INTO images (image) VALUES ('{word}')"
                # Execute the search query
                cursor = conn.cursor()
                cursor.execute(query)
            else:
                query = f"INSERT INTO images (image) VALUES ('{word + str(0)}')"
                word = word+str(0)
                # Execute the search query
                cursor = conn.cursor()
                cursor.execute(query)
            conn.commit()

        

        
        import torch
        from PIL import Image
        from tqdm.auto import tqdm
        from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
        from point_e.diffusion.sampler import PointCloudSampler
        from point_e.models.download import load_checkpoint
        from point_e.models.configs import MODEL_CONFIGS, model_from_config
        from point_e.util.plotting import plot_point_cloud

        """### Models"""

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print('creating base model...')
        base_name = 'base40M' # use base300M or base1B for better results
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
            guidance_scale=[3.0, 3.0],
        )   

        import sys

        # Read the first command-line argument
        #prompt = file

        realPrompt = word
        img = Image.open(files)
        #print(realPrompt)
        # Produce a sample from the model.
        samples = None
        for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(images=[img]))):
            samples = x
            break
        print('sampled')
        # Plot the sample.

        pc = sampler.output_to_point_clouds(samples)[0]

        #fig = plot_point_cloud(pc, grid_size=3, fixed_bounds=((-0.75, -0.75, -0.75),(0.75, 0.75, 0.75)))

        from point_e.util.pc_to_mesh import marching_cubes_mesh
        import os
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
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
        #print('inserted')

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

        #with open(file_path, 'wb') as f:
            #f.write(binary_data)

        import numpy as np
        import stl
        import tempfile

        cursor = conn.cursor()
        cursor.execute(f"SELECT data FROM {realPrompt}files WHERE name = '{realPrompt}.stl'")
        stl_file = cursor.fetchone()[0]

        # Convert the memory view to a bytes object
        stl_bytes = bytes(stl_file)
        #print(stl_bytes)

        # Create a temporary file and write the stl_bytes object to it
        with tempfile.NamedTemporaryFile(mode='w+b', suffix='.stl', delete=False) as fp:
            fp.write(stl_bytes)
            stl_file = fp.name

        mesh = stl.mesh.Mesh.from_file(stl_file)

        vertices = mesh.vectors

        #print((vertices))
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
        return binary_data
    except ValueError:
        return "invalid input"


from flask import send_file
realBinaryData = b""
realPrompt2 = ""
@app.route('/text2image/download/')
def download_file(data, name):
    #print(realPrompt2)
    #print(realBinaryData)
    # binary_data contains the binary data for the file you want to download
    print(data)
    print(name)
    response = send_file(data, attachment_filename=name + '.stl', as_attachment=True)
    return response

@app.route("/text2image")
def text2image():
    word = request.args.get("word", "")
    if word:
        fileName = file_from(word)
    else:
        fileName = ""
    return render_template('text2image.html', fileName=fileName)
"""@app.route("/text2image?word=<word>")
def future():
    word = request.args.get("word", "")
    if word:
        fileName = file_from(word)
    else:
        fileName = ""
    return render_template('text2image.html', fileName=fileName)"""


def file_from(word):
    """Convert Celsius to Fahrenheit degrees."""
    try:
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
        print("connected")
        query = f"SELECT * FROM words"
        print('got word')

    # Execute the insert query
        cursor = conn.cursor()
        cursor.execute(query)
        
        # Fetch the results
        results = cursor.fetchall()
        entered = False
        temp = []
        for row in results:
            name = row
            temp.append(name)
            entered = True
        if not entered:
            query = f"INSERT INTO words (word) VALUES ('{word + str(0)}')"
            word = word+str(0)
            # Execute the search query
            cursor = conn.cursor()
            cursor.execute(query)
        else:
            enteredAgain = False
            for i in temp:
                war = i[1]
                if war[:len(i[1])-1] == word:
                    enteredAgain = True
                    name = i[1]
                    print(name)
            if enteredAgain:
                word = word+str(int(name[-1])+1)
                query = f"INSERT INTO words (word) VALUES ('{word}')"
                # Execute the search query
                cursor = conn.cursor()
                cursor.execute(query)
            else:
                query = f"INSERT INTO words (word) VALUES ('{word + str(0)}')"
                word = word+str(0)
                # Execute the search query
                cursor = conn.cursor()
                cursor.execute(query)
            
            

        conn.commit()

        


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
        prompt = word

        realPrompt = prompt
        realPrompt = realPrompt.replace(' ', '_')
        #print(realPrompt)
        # Produce a sample from the model.
        samples = None
        for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt]))):
            samples = x
            break
        print('sampled')
        # Plot the sample.

        pc = sampler.output_to_point_clouds(samples)[0]

        #fig = plot_point_cloud(pc, grid_size=3, fixed_bounds=((-0.75, -0.75, -0.75),(0.75, 0.75, 0.75)))

        from point_e.util.pc_to_mesh import marching_cubes_mesh
        import os
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
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
        #print('inserted')

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
        #print(stl_bytes)

        # Create a temporary file and write the stl_bytes object to it
        with tempfile.NamedTemporaryFile(mode='w+b', suffix='.stl', delete=False) as fp:
            fp.write(stl_bytes)
            stl_file = fp.name

        mesh = stl.mesh.Mesh.from_file(stl_file)

        vertices = mesh.vectors

        #print((vertices))
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
        #realBinaryData = binary_data
        #realPrompt2 = realPrompt
        return binary_data, realPrompt



    except ValueError:
        return "invalid input"

if __name__ == "__main__":
    app.run(debug=True)

<p align="center">
LogIntelligence - Analyze your ssh logs file.
</p>

---

### Image from dockerfile

- If you are using a machine other than a Jetson, you can just run the following commands

- If you are using a Jetson, you should uncomment each Dockerfile with the dustynv image.

### Commands to execute the application

To build the image and run it:

```bash
docker compose up --build
```

To restart the application:

```bash
docker compose restart
```

To clean and rebuild:

```bash
./build.sh
```

### Getting Started

After running the docker compose, a fastapi client should be opened on 0.0.0.0/8000 on Jetson or 0.0.0.0/12312 on Ubuntu.

The first API call demo can be used to generate some answer from the essay of Paul Graham.

Then, given the path and the file_name of the ssh logs that need to be analyze, please use the next functions (database, analyzer and everything).

First, you need to create a database, so please call the database before the analyzer.

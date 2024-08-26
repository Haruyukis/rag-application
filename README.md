<p align="center">
LogIntelligence - Analyze your ssh logs file.
</p>

---

### Image from dockerfile

- If you are using a machine other than a Jetson, you should uncomment each Dockerfile corresponding to the ubuntu20.04

- If you are using a Jetson, you can just run the following command.

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

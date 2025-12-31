class JobService:
    def create_job(self, keywords, target):
        print(f"Creating job for {target}")
        return {"id": "job_1", "keywords": keywords, "target": target}

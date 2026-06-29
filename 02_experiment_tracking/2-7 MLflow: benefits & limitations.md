## 2.7 MLflow: benefits, limitations, and alternatives
Benefits of remote tracking
* Share experiments with other data scientists
* Collaborate with others to build and deploy models
* Provide more visibility of the efforts

NB Be aware of
* Security: (Make sure you restrict the access to the server)
* Scalability: If you have lots of experiments and users you may need to think about how to scale this up
    * Deploy MLflow on AWS Fargate. e.g. [This medium article](https://vaibhavsatpathy.medium.com/mlops-deployment-into-aws-fargate-i-bd612af5dd7a)
    * [MLflow at Company Scale](https://www.slideshare.net/databricks/mlflow-at-company-scale-239587085)
* Isolation
    * Naming conventions
    * Restrict access to artifacts

**ML Flow limitations**

* `Authentication and Users`: The free tier of MLFlow doesn't provide any sort of authentication. (Databricks does though)
* `Data Versioning`: MLflow doesn't provide data versioning. There are some work arounds like controling paths etc.
* `Model/Data Monitoring & Alerting`: This is out of the scope of MLflow, but you can calculate some variables to set up as monitors.

**Alternatives**
* [Neptune](https://neptune.ai/)
* [Comet](https://www.comet.com/site/)
* [Weights and Biases](https://wandb.ai/site)
Free as an individual, pay if there is a team.

For a summary of the different tools available see Neptune.Ais [blog post](https://neptune.ai/blog/best-ml-experiment-tracking-tools)

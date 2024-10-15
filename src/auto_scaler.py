import kubernetes
from kubernetes import client, config

class AutoScaler:
    def __init__(self):
        config.load_incluster_config()
        self.apps_v1 = client.AppsV1Api()
        self.autoscaling_v2 = client.AutoscalingV2Api()

    def scale_deployment(self, name, namespace, replicas):
        body = {
            "spec": {
                "replicas": replicas
            }
        }
        self.apps_v1.patch_namespaced_deployment_scale(name, namespace, body)

    def update_hpa(self, name, namespace, min_replicas, max_replicas, target_cpu_utilization):
        body = client.V2HorizontalPodAutoscaler(
            api_version="autoscaling/v2",
            kind="HorizontalPodAutoscaler",
            metadata=client.V1ObjectMeta(name=name, namespace=namespace),
            spec=client.V2HorizontalPodAutoscalerSpec(
                scale_target_ref=client.V2CrossVersionObjectReference(
                    api_version="apps/v1",
                    kind="Deployment",
                    name=name
                ),
                min_replicas=min_replicas,
                max_replicas=max_replicas,
                metrics=[
                    client.V2MetricSpec(
                        type="Resource",
                        resource=client.V2ResourceMetricSource(
                            name="cpu",
                            target=client.V2MetricTarget(
                                type="Utilization",
                                average_utilization=target_cpu_utilization
                            )
                        )
                    )
                ]
            )
        )
        self.autoscaling_v2.replace_namespaced_horizontal_pod_autoscaler(name, namespace, body)
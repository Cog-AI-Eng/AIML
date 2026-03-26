# Guide 5: Setup CloudWatch Alerts

Navigate to CloudWatch to find Model Monitor metrics, create an alarm that triggers when data quality violations are detected, and configure SNS email notifications so the FraudShield team is alerted automatically.

---

## Steps

### Step 1 -- Navigate to CloudWatch Metrics

1. Open the **AWS Management Console** and navigate to **Amazon CloudWatch**.
2. In the left sidebar, select **Metrics** then **All metrics**.
3. In the metrics search bar, type `sagemaker` and look for the namespace **aws/sagemaker/Endpoints/data-metrics** or **/aws/sagemaker/ModelMonitor**.
4. If Model Monitor metrics are not yet visible, it may be because no monitoring execution has completed with violations. Proceed to create the alarm referencing the expected metric name.

---

### Step 2 -- Explore Available Model Monitor Metrics

1. Under the SageMaker metrics namespace, browse for metrics related to your endpoint.
2. Look for metrics such as:
   - `data_quality/violations_count` -- Number of constraint violations detected per execution.
   - `data_quality/features_with_violations` -- Number of features that triggered violations.
3. Select a metric and click **Graphed metrics** to view its historical values.
4. Note the metric name and dimensions (endpoint name, schedule name) for the alarm configuration.

---

### Step 3 -- Create a CloudWatch Alarm

1. In the left sidebar, select **Alarms** then **All alarms**.
2. Click **Create alarm**.
3. Click **Select metric**.
4. Navigate to the SageMaker/ModelMonitor namespace and select the `data_quality/violations_count` metric for your endpoint. If the metric does not appear in the console yet, you can create it manually:
   - **Namespace:** `aws/sagemaker/Endpoints/data-metrics`
   - **Metric name:** `data_quality_violations_count`
   - **Dimensions:** EndpointName = `<your-endpoint-name>`
5. Click **Select metric**.
6. Under **Conditions**:
   - **Threshold type:** Select **Static**.
   - **Whenever violations_count is:** Select **Greater than**.
   - **Threshold value:** Enter `0`.
   - This triggers the alarm whenever any violation is detected.
7. Under **Period**, select `1 hour` to align with the monitoring schedule.
8. Click **Next**.

---

### Step 4 -- Configure SNS Notification

1. Under **Notification**, select **In alarm**.
2. Choose **Create new topic**.
3. For **Topic name**, enter `ASM-FraudShield-MonitorAlerts`.
4. For **Email endpoint**, enter your email address.
5. Click **Create topic**.
6. Click **Next**.
7. For **Alarm name**, enter `ASM-FraudShield-DataQuality-Alarm`.
8. For **Alarm description**, enter: `Triggers when FraudShield data quality monitoring detects constraint violations.`
9. Click **Next**, review the configuration, and click **Create alarm**.
10. Check your email inbox and **confirm** the SNS subscription.

---

### Step 5 -- Verify the Alarm

1. Return to **CloudWatch Alarms** and locate `ASM-FraudShield-DataQuality-Alarm`.
2. The alarm state will initially be **Insufficient data** (no monitoring data has been evaluated yet against the threshold).
3. After the next monitoring execution completes, the alarm will transition to either:
   - **OK** -- No violations detected.
   - **ALARM** -- Violations exceeded the threshold.
4. Click on the alarm name to view its history and state transitions.
5. If the alarm enters ALARM state, verify you received an email notification at the configured address.

---

## Presentation Checkpoint
Be prepared to show:
- The CloudWatch alarm configuration page showing the metric, threshold, and period
- The SNS topic with a confirmed email subscription
- The alarm state (OK, ALARM, or Insufficient data) on the alarms dashboard

---

## Key Concepts
- **CloudWatch Metrics from Model Monitor:** SageMaker Model Monitor publishes metrics to CloudWatch after each monitoring execution, including violation counts and feature-level statistics.
- **Static Threshold Alarm:** Fires when a metric crosses a fixed value. Setting the threshold to 0 for violations means any violation triggers the alarm.
- **SNS Integration:** CloudWatch alarms can send notifications via SNS to email, SMS, Lambda functions, or other subscribers, enabling automated alerting workflows.
- **Alarm States:** CloudWatch alarms have three states: OK (metric within threshold), ALARM (metric breached threshold), and INSUFFICIENT_DATA (not enough data points to evaluate).

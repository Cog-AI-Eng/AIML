# Guide 5: Build a CloudWatch Dashboard

Create a CloudWatch dashboard with four or more widgets that provide operational visibility into the FraudShield inference fleet. The dashboard will display endpoint invocations, latency percentiles, CPU utilization, and error rates at a glance.

---

## Steps

### Step 1 -- Create a New Dashboard

1. Open the **AWS Management Console** and navigate to **Amazon CloudWatch**.
2. In the left sidebar, select **Dashboards**.
3. Click **Create dashboard**.
4. For **Dashboard name**, enter `ASM-FraudShield-Operations`.
5. Click **Create dashboard**.
6. A dialog will ask you to add your first widget. Select **Line** chart and click **Next**.

---

### Step 2 -- Add an Invocations Widget

1. In the metric selection screen, search for `AWS/SageMaker`.
2. Select the **Endpoint** dimension group.
3. Find and select the `Invocations` metric for your FraudShield endpoint and variant.
4. Set the **Period** to `1 minute` and **Statistic** to `Sum`.
5. Click **Create widget**.
6. The widget title defaults to the metric name. Click the pencil icon on the widget title and rename it to `Endpoint Invocations (per minute)`.

---

### Step 3 -- Add a Latency Widget (p50 and p99)

1. Click **Add widget** at the top of the dashboard.
2. Select **Line** chart and click **Next**.
3. Search for `AWS/SageMaker` and select the **Endpoint** dimension group.
4. Select the `ModelLatency` metric for your endpoint.
5. Click **Graphed metrics** tab.
6. Add the metric twice with different statistics:
   - First line: **Statistic** = `p50`, **Label** = `Latency p50`
   - Second line: **Statistic** = `p99`, **Label** = `Latency p99`
7. Set **Period** to `1 minute` for both.
8. Click **Create widget**.
9. Rename the widget to `Model Latency (p50 / p99)`.

---

### Step 4 -- Add a CPU Utilization Widget

1. Click **Add widget** and select **Line** chart.
2. Search for `AWS/SageMaker` and select the **Endpoint** dimension group.
3. Select the `CPUUtilization` metric for your endpoint variant.
4. Set **Period** to `1 minute` and **Statistic** to `Average`.
5. Click **Create widget**.
6. Rename the widget to `Endpoint CPU Utilization (%)`.

---

### Step 5 -- Add an Errors Widget (4xx and 5xx)

1. Click **Add widget** and select **Number** (single value) chart.
2. Search for `AWS/SageMaker` and select the **Endpoint** dimension group.
3. Select the `Invocation4XXErrors` metric for your endpoint.
4. Add a second metric: `Invocation5XXErrors` for the same endpoint.
5. Set **Period** to `1 hour` and **Statistic** to `Sum` for both.
6. Click **Create widget**.
7. Rename the widget to `Invocation Errors (4xx / 5xx)`.

---

### Step 6 -- Add an Alarm Status Widget

1. Click **Add widget** and select **Alarm status**.
2. In the alarm selector, choose any alarms you created in previous labs (e.g., the data quality alarm from Module 5 or the auto-scaling alarms from Guide 3).
3. If no alarms exist, create a simple alarm first:
   - Go to **Alarms** and create an alarm on `Invocation5XXErrors > 0` for your endpoint.
   - Name it `ASM-FraudShield-5xx-Alarm`.
   - Return to the dashboard and add this alarm to the widget.
4. Click **Create widget**.
5. Rename the widget to `Alarm Status`.

---

### Step 7 -- Arrange and Save the Dashboard

1. Drag and resize the widgets to create a clean layout:
   - Top row: Invocations (left), Latency (right)
   - Bottom row: CPU Utilization (left), Errors (center), Alarm Status (right)
2. Click **Save dashboard** in the top right corner.
3. Verify the dashboard loads correctly by refreshing the page.
4. Set the time range selector to **1h** or **3h** to see recent activity.

---

## Presentation Checkpoint
Be prepared to show:
- The completed dashboard with all four or more widgets visible
- The Invocations widget showing request volume
- The Latency widget with both p50 and p99 lines
- The Alarm Status widget showing alarm states

---

## Key Concepts
- **CloudWatch Dashboard:** A customizable page of widgets that display metrics, alarms, and logs from across AWS services. Dashboards provide a single pane of glass for operational monitoring.
- **Latency Percentiles:** p50 (median) shows typical latency; p99 shows the latency experienced by the slowest 1% of requests. Monitoring both reveals whether tail latency is problematic.
- **Alarm Status Widget:** Displays the state of CloudWatch alarms directly on the dashboard, providing immediate visual feedback when something goes wrong.
- **Dashboard Cost:** The first 3 dashboards are free. Each additional dashboard costs $3/month. Widgets within a dashboard are free.

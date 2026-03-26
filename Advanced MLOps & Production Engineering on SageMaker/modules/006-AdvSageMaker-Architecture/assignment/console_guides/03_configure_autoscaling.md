# Guide 3: Configure Auto-Scaling

Navigate to your FraudShield endpoint's auto-scaling configuration and create a target-tracking scaling policy based on invocations per instance. This ensures the endpoint automatically scales out during traffic spikes and scales in during quiet periods.

---

## Steps

### Step 1 -- Navigate to the Endpoint

1. Open the **AWS Management Console** and navigate to **Amazon SageMaker**.
2. In the left sidebar under **Inference**, select **Endpoints**.
3. Click on your active FraudShield endpoint to open its detail page.
4. Confirm the endpoint is in **InService** status with at least 1 instance running.
5. Note the **Variant name** (usually `AllTraffic`) -- you will need this for the scaling configuration.

---

### Step 2 -- Access the Auto-Scaling Configuration

1. On the endpoint detail page, scroll down to the **Endpoint runtime settings** section.
2. Select the production variant (e.g., `AllTraffic`).
3. Click **Configure auto scaling** (or look for the auto-scaling option in the variant settings).

---

### Step 3 -- Set Instance Count Boundaries

1. In the auto-scaling configuration dialog:
   - **Minimum instance count:** Enter `1`. The endpoint will always have at least 1 instance running.
   - **Maximum instance count:** Enter `3`. During peak traffic, the endpoint can scale up to 3 instances.
2. These boundaries prevent runaway scaling while ensuring minimum availability.

---

### Step 4 -- Create a Target-Tracking Scaling Policy

1. Under **Scaling policy**, select **Target tracking**.
2. For **Policy name**, enter `ASM-FraudShield-ScalingPolicy`.
3. For **Target metric**, select **SageMakerVariantInvocationsPerInstance**.
4. For **Target value**, enter `100`. This means the scaling policy will attempt to maintain approximately 100 invocations per instance. When the average exceeds this, SageMaker adds instances; when it drops below, instances are removed.
5. Under **Scale-in cool down (seconds)**, enter `300` (5 minutes). This prevents premature scale-in after a scale-out event.
6. Under **Scale-out cool down (seconds)**, enter `60` (1 minute). This allows rapid response to traffic spikes.
7. Click **Save** or **Apply**.

---

### Step 5 -- Verify the Scaling Policy

1. Return to the endpoint detail page.
2. Under **Endpoint runtime settings**, verify the auto-scaling configuration now shows:
   - Min instances: 1
   - Max instances: 3
   - Scaling policy: Target tracking on InvocationsPerInstance
3. Navigate to **Application Auto Scaling** in the AWS console (or check under **CloudWatch > Alarms**).
4. You should see two auto-generated CloudWatch alarms:
   - A high alarm that triggers scale-out when invocations per instance exceed the target.
   - A low alarm that triggers scale-in when invocations per instance drop below the target.
5. Note the alarm names for reference.

---

### Step 6 -- Test the Scaling Behavior (Optional)

1. Open **AWS CloudShell** and send a burst of requests to simulate increased traffic:
   ```bash
   for i in $(seq 1 500); do
     aws sagemaker-runtime invoke-endpoint \
       --endpoint-name <your-endpoint-name> \
       --content-type text/csv \
       --body "50.0,1,0,1,234.56,2,0.85,1200,3" \
       --region us-east-1 \
       /dev/null &
   done
   wait
   ```
2. Monitor the endpoint detail page to see if the instance count increases.
3. After traffic subsides, wait 5+ minutes to observe scale-in behavior.

---

## Presentation Checkpoint
Be prepared to show:
- The auto-scaling configuration showing min/max instance counts
- The target-tracking policy with the InvocationsPerInstance metric and target value of 100
- The auto-generated CloudWatch alarms for scale-out and scale-in

---

## Key Concepts
- **Target-Tracking Scaling:** A policy that automatically adjusts instance count to keep a chosen metric near a target value. SageMaker handles the math of when to add or remove instances.
- **InvocationsPerInstance:** The average number of invocations each instance receives per minute. It is the most common scaling metric for SageMaker endpoints.
- **Cool Down Period:** A waiting period after a scaling action during which no further scaling occurs. Scale-in cool down prevents oscillation; scale-out cool down prevents overreaction to brief spikes.
- **Min/Max Instances:** Guard rails that prevent the endpoint from scaling below a minimum (ensuring availability) or above a maximum (controlling costs).

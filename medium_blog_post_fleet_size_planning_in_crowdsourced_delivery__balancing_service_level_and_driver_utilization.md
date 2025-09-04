Fleet Size Planning in Crowdsourced Delivery: Balancing Service Level and Driver Utilization
============================================================================================

![captionless image](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*0h0EKhtealjtWjjk-wByHQ.png)

[Reference](https://medium.com/@sahil.bhatt/fleet-size-planning-in-crowdsourced-delivery-a-multi-objective-optimization-approach-e868f621717e)

by [Sahil Bhatt](https://medium.com/@sahil.bhatt?source=post_page---byline--e868f621717e---------------------------------------)


10 hours ago




Research Context and Motivation
-------------------------------

Crowdsourced delivery platforms face increasing regulatory pressure to improve driver working conditions while maintaining service quality. Cities like New York, Ontario, and Minnesota have implemented minimum wage guarantees and utilization standards, with some jurisdictions capping platform fleet sizes. This research, under review at Omega: The International Journal of Management Science, addresses how platforms can optimize fleet sizes to balance competing objectives: service level (percentage of orders fulfilled) and driver utilization (percentage of time drivers are active).

The Technical Challenge
-----------------------

The problem presents several complexities:

1.  **Decision-dependent uncertainty**: Driver availability depends on fleet size decisions. This represents a form of endogenous uncertainty where first-stage decisions affect second-stage probability distributions.
2.  **Computational intractability**: With P periods and up to x̄ drivers per period, the solution space contains x̄^P possible configurations.
3.  **Stochastic elements**: Both driver arrivals (binomial distribution) and order arrivals (Poisson process) are probabilistic.
4.  **Multi-period dynamics**: Fleet sizes must be determined for multiple time periods (e.g., hourly) throughout the operating day.

Mathematical Formulation
------------------------

The research proposes a two-stage stochastic optimization model:

First Stage: Tactical Fleet Sizing
----------------------------------

Second Stage: Operational MDP
-----------------------------

The operational problem is modeled as a Markov Decision Process with:

*   **State space**: Driver attributes (location, availability, utilization), order attributes (origin, destination, time window), platform metrics
*   **Action space**: Driver-order matching decisions
*   **Transition function**: Captures stochastic driver and order arrivals
*   **Reward function**: Immediate profit minus penalties for unmet targets

Solution Methodology: Value Function Approximation
--------------------------------------------------

Given the computational complexity, the research employs a Value Function Approximation (VFA) algorithm:

![captionless image](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*l3Sh4tusGi-WyWW6hgNulA.png)

Key Findings: The Trade-off Between Service Level and Driver Utilization
------------------------------------------------------------------------

The research tested six weight configurations to understand the trade-off between service level and driver utilization:

Weight w = 1.0 (Considers Service Level Only, Disregarding Driver Utilization)
------------------------------------------------------------------------------

*   **Aggregate Fleet Size**: 1,078 drivers
*   **Service Level**: 97% ± 1%
*   **Driver Utilization**: 37% ± 31%
*   **Drivers Meeting 80% Target**: 13%
*   **Platform Daily Profit**: $14,600
*   **Average Idle Time**: 40.4 minutes

Weight w = 0.8 (Balanced Approach)
----------------------------------

*   **Aggregate Fleet Size**: 381 drivers
*   **Service Level**: 98% ± 2%
*   **Driver Utilization**: 93% ± 10%
*   **Drivers Meeting 80% Target**: 83%
*   **Platform Daily Profit**: $14,000
*   **Average Idle Time**: 4.48 minutes

Weight w = 0.5 (Equal Balance)
------------------------------

*   **Aggregate Fleet Size**: 376 drivers
*   **Service Level**: 97% ± 2%
*   **Driver Utilization**: 93% ± 10%
*   **Drivers Meeting 80% Target**: 83%
*   **Platform Daily Profit**: $14,100
*   **Average Idle Time**: 4.51 minutes

Weight w = 0 (Utilization Only)
-------------------------------

*   **Aggregate Fleet Size**: 88 drivers
*   **Service Level**: 29% ± 2%
*   **Driver Utilization**: 100% ± 0%
*   **Platform Daily Profit**: $6,030
*   **Average Idle Time**: 0 minutes

Critical Insights
-----------------

1. Service-Utilization Trade-off Analysis
------------------------------------------

The results demonstrate that the perceived trade-off between service and utilization is less severe than industry practice suggests. Moving from w = 1.0 to w = 0.8 achieves:

*   65% fleet reduction
*   1% service level improvement
*   56 percentage point utilization increase
*   Only 4% profit reduction

2. Driver Compliance Sensitivity
---------------------------------

Driver reliability, as modeled through the probability of driver arrival (qₚ) , significantly impacts fleet size decisions. A higher qₚ means drivers are more likely to log in during their scheduled periods, which allows the platform to maintain service levels with a smaller fleet. Conversely, when qₚ is low and driver availability is uncertain, the platform must overstaff to avoid order delays or unfulfilled requests, leading to higher operational costs and lower driver utilization.

![captionless image](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*YAQ0XGOSR8Rzc0KoYILojg.png)

**Key Insight**
Platforms benefit the most from predictable driver behavior. When drivers reliably show up for scheduled periods, platforms can cut fleet sizes by over 60%, achieving high utilization and strong service levels , which benefits both drivers and the business. But when driver reliability is low, platforms must overstaff to maintain service quality, which raises costs and lowers utilization.

3. Temporal Flexibility Benefits
---------------------------------

Comparing dynamic versus constant fleet sizes for w = 0.5:

**Dynamic Policy**

*   Service Level: 97%
*   Utilization: 94%
*   Total Drivers: 376

**Constant Policy**

*   Service Level: 92.8%
*   Utilization: 96.7%
*   Total Drivers: 336

Dynamic adjustment enables meeting both targets simultaneously.

4. Empty Distance Trade-off
----------------------------

Average pickup distance increases with balanced objectives:

*   w = 1.0: 1.93 km
*   w = 0.8: 2.56 km (+33%)

This reflects fewer available drivers but is offset by significant improvements in driver utilization.

Chicago Dataset Validation
--------------------------

Using Chicago ridehailing data (2018–2022), focusing on downtown areas (community areas 8, 32, 33):

Results at Different Weights
----------------------------

**w = 0.2**

*   Fleet Size: 474
*   Service Level: 81%
*   Utilization: 99%
*   Daily Profit: $12,820

**w = 0.5**

*   Fleet Size: 703
*   Service Level: 99%
*   Utilization: 97%
*   Daily Profit: $10,120

**w = 0.8**

*   Fleet Size: 685
*   Service Level: 97%
*   Utilization: 96%
*   Daily Profit: $10,540

**w = 1.0**

*   Fleet Size: 1,158
*   Service Level: 96%
*   Utilization: 66%
*   Daily Profit: $10,330

Notable: Maximum profit occurs at w = 0.2, not w = 1.0, indicating some orders have negative contribution margins.

Computational Performance
-------------------------

*   **Average optimization time per epoch**: 0.02 seconds
*   **Total PCFA time per iteration**: 36–75 seconds
*   **Convergence**: ~400 iterations for utilization, ~200 for service level
*   **Monte Carlo scenarios**: 10 provides good accuracy/efficiency balance

Sensitivity Analysis: Impact of Planning Periods
------------------------------------------------

The research highlights the importance of **temporal granularity**, that is, how frequently platforms revisit fleet size decisions throughout the day. Shorter planning periods give platforms the flexibility to adapt to fluctuating demand:

*   **4 periods/day (4 hours each)** → Service 93%, Utilization 84%
*   **8 periods/day (2 hours each)** → Service 97%, Utilization 90%
*   **16 periods/day (1 hour each)** → Service 97%, Utilization 93%
*   **32+ periods/day (30 min or less)** → Service and utilization both ~98%

**Key Insight**
More frequent fleet planning allows platforms to **respond quickly to demand surges** while **avoiding excess idle capacity**. However, finer granularity comes at the cost of **higher computational effort**. For most delivery platforms, **hourly updates** provide near-optimal service and utilization without heavy processing costs.

Implementation Considerations
-----------------------------

For Platforms
-------------

1.  **Regulatory Compliance**: Meet minimum utilization standards while maintaining service
2.  **Driver Relations**: Higher utilization improves driver satisfaction and retention
3.  **Operational Efficiency**: Reduce recruitment and onboarding costs

For Policy Makers
-----------------

1.  **Evidence-based Standards**: 80% utilization and 95% service are achievable simultaneously
2.  **Fleet Size Caps**: Can improve driver welfare without harming service
3.  **Compliance Importance**: Driver reliability dramatically affects achievable outcomes

Limitations and Extensions
--------------------------

Current Limitations
-------------------

*   Single service region assumption
*   Deterministic service times
*   Known demand distributions
*   Homogeneous driver capabilities

Future Research
---------------

*   Multi-region coordination
*   Joint optimization with dynamic pricing
*   Driver preference learning
*   Deep reinforcement learning for larger state spaces

Technical Implementation Details
--------------------------------

Parametric Cost Function Approximation
--------------------------------------

![captionless image](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*pVG3o4DDUJHCYzKaYnYNIA.png)

Conclusion
----------

This research challenges conventional operational approaches in the gig economy. By properly formulating and solving the fleet size planning problem with decision-dependent uncertainty, platforms can achieve substantial fleet reductions while improving both service levels and driver utilization. The key finding is that optimizing across multiple objectives simultaneously produces superior outcomes compared to focusing on a single objective.

The 65% fleet reduction with w = 0.8 represents a feasible path toward sustainable gig economy operations that benefit all stakeholders without requiring fundamental business model changes.

**Technical Note**: The MDP formulation includes comprehensive state tracking (driver utilization, time in system, order time windows) solved via parametric cost function approximation with rolling horizon implementation. The decision-dependent uncertainty is handled through iterative value function updates that learn the relationship between fleet decisions and operational outcomes.

**Data Availability**: Synthetic instances and Chicago dataset preprocessing code available upon request.

**Keywords**: Fleet Size Optimization, Stochastic Optimization, Value Function Approximation, Markov Decision Process, Gig Economy, Operations Research
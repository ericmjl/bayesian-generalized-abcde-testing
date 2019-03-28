# Generalized Bayesian A/B[/C/D/E...] Testing

Speaker: Eric J. Ma
Affiliation: Novartis Institutes for Biomedical Research

## About Me

3 logos:

- MIT
- Novartis
- Python

## QR Code & URL

<!-- Put QR code here. Include QR code generator as part of Travis build script. -->

## Take-Home Messages

- You can do comparisons beyond just group "A" and group "B". You can add groups "C", "D", "E" and more.
- Using a probabilistic programming language, such as PyMC3, lets you do this in a principled fashion with a wonderful API.

## Comparison is a cornerstone of scientific inquiry

<!-- Background should be Hans Rosling video clip. -->

> When I see a lonely number in a news report, it always triggers an alarm: ***What should this lonely number be compared to?*** (Hans Rosling, "Factfulness")

## Bayesian Estimation: Comparison of 2 Groups

<!-- What are you most used to seeing? T-test -->

![](./images/best.png)

Let's say we're measuring the effect of an intervention on IQ.

Business question at hand:

> Does the intervention work?

This gets translated into the following statistical inference question, which we set up as our proxy for answering business question:

> Given the data, **can we conclude that $\mu_{1}$ and $\mu_{2}$ different?**

You probably heard about the t-test. Let's make clear what the assumptions of the t-test are.

We:

- assume that one of the groups is a "baseline" or "null hypothesis" group
- assume that our data are t-distributed
- assume that our mean is normally distributed, by "something something CLT"
- make an unclear assumption about how our variance and degrees of freedom are distributed

If we make all of these assumptions plain and clear, we arrive at Bayesian estimation.

Let's very quickly look at the model +

### Code

```python
with pm.Model() as model:
    obs1 = pm.Data('obs1', df.query('group == 1').values)
    obs2 = pm.Data('obs2', df.query('group == 2').values)

    mu1 = pm.Normal('mu1', mu=0, sd=10)
    mu2 = pm.Normal('mu2', mu=0, sd=10)

    sigma1 = pm.HalfCauchy('sigma1')
    sigma2 = pm.HalfCauchy('sigma2')

    nu = pm.Exponential('nu', lam=1/29.)

    like1 = pm.StudentT('like1', mu=mu1, sigma=sigma1, nu=nu, observed=obs1)
    like2 = pm.StudentT('like2', mu=mu1, sigma=sigma1, nu=nu, observed=obs2)
```

In Bayesian estimation, we:

- estimate the parameters for each group, and then
- compare them

There's no reason why we should stop at 2 groups.

## Bayesian Estimation: Extension to >3 Groups

![](./images/best-3-groups.png)

```python
n_groups = ...
with pm.Model() as model:
    # Observations
    obs = pm.Data('observations', ...)

    # Model Parameters
    mu = pm.Normal('mu', mu=0, sd=10, shape=n_groups)
    sigma = pm.HalfCauchy('sigma', shape=n_groups)
    nu = pm.Exponential('nu', lam=1/29.)

    # Likelihood
    like = pm.StudentT(
        'like',
        mu=mu[df['groups']],
        sigma=sigma[df['groups']],
        nu=nu,
        observed=obs
    )
```

- **Point 1:** With Bayesian methods, we can go beyond two group comparisons.
- **Point 2:** With PyMC3, extending Bayesian estimation beyond two groups is trivial.

## Side Note: Flexibility

If our estimation problem didn't deal with continuous outputs, all we would have to do is to replace the likelihood with a different distribution.

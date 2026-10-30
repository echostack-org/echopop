(semivariogram_eq)=
# Theoretical semivariogram equations

## Parameter definitions

| Parameter            | Symbol | Description                | Practical Limits      |
|----------------------|--------|----------------------------|----------------------|
| Nugget               | $C_0$  | Value at zero lag;<br>measurement error<br>micro-scale variation. | $C_0 \geq 0$         |
| Sill                 | $C$    | Total variance at large lag;<br>plateau for bounded models. | $C > C_0$            |
| Partial sill         | $C_1$  | Sill minus nugget<br>($C_1 = C - C_0$). | $C_1 \geq 0$         |
| Range                | $a$, $L$| Lag where sill is reached ($a$)<br>or a fraction of it ($L$). | $a > 0$, $L > 0$     |
| Hole effect range    | $b$    | Controls periodicity<br>in "hole effect" models. | $b \geq 0$              |
| Decay          | $\alpha$    | Exponent for decay<br>usually $0 < \alpha \leq 2$. | $\alpha > 0$              |
| Shape      | $\beta$| Polynomial decay control<br>in quadratic/power models. | $\beta > 0$          |
| Smoothnes | $\nu$  | Differentiability in Matérn<br>model. | $\nu > 0$            |
| Power | $p$ | Exponent for the power law model. | $0 < p < 2$ |

## Special Functions and Symbols

| Symbol / Function         | Definition                                                           |
|--------------------------|-----------------------------------------------------------------------|
| $\text{J}_0(x)$ | Cylindrical Bessel function of the first kind of order 0: <br> $\text{J}_0(x) = \frac{1}{\pi} \int_0^\pi \cos(x \sin \theta) d\theta$, <br> where $x$ is the argument and $\theta$ is the integration variable. |
| $\text{K}_\nu(x)$ | Modified cylindrical Bessel function of the second kind of order $\nu$: <br> $\text{K}_\nu(x) = \int_0^\infty e^{-x \cosh t} \cosh(\nu t) dt$, <br> where $x$ is the argument, $t$ is the integration variable, and $\cosh$ is the hyperbolic cosine. |
| $\Gamma(\nu)$            | Gamma function: $\Gamma(\nu) = \int_0^\infty t^{\nu-1} e^{-t} dt$                           |
| $\Omega$                 | Angular distance between two points on a sphere used in the Legendre semivariogram model [Legendre semivariogram model](eq-legendre).                        |

---

## Models

### Circular

$$
\gamma(h) =
\begin{cases}
C_1 \left[ 1 - \frac{2}{\pi} \cos^{-1}\left(\frac{h}{a}\right) + \frac{2h}{\pi a} \sqrt{1 - \frac{h^2}{a^2}} \right], & h \leq a \\
C_1, & h > a
\end{cases}
\label{eq:circular}
$$

The circular model describes a spatial process with a finite range, where the semivariogram rises sharply and then flattens at the sill. The parameters $C_1$ and $a$ represent the sill and range, respectively. This model is conditionally negative definite in one and two dimensions, but not in three. It is useful for phenomena where spatial correlation drops off tightly as the range is approached. {cite:p}`chiles_geostatistics_2012`

---

### Cubic

$$
\gamma(h) =
\begin{cases}
C_0 + C_1 \left( 7\left(\frac{h}{a}\right)^2 - \frac{35}{4}\left(\frac{h}{a}\right)^3 + \frac{7}{2}\left(\frac{h}{a}\right)^5 - \frac{3}{4}\left(\frac{h}{a}\right)^7 \right), & h \leq a \\
C_0 + C_1, & h > a
\label{eq:cubic}
\end{cases}
$$

The cubic model produces a very smooth transition from the origin to the sill, with continuous derivatives up to higher orders. This model is suitable for spatial processes that are expected to be highly regular and smooth, such as certain environmental or physical phenomena. {cite:p}`chiles_geostatistics_2012, montero_2015`

---

### Exponential

$$
\gamma(h) = C_0 + C_1 \left( 1 - e^{-h/a} \right)
\label{eq:exponential}
$$

The exponential model describes a process where spatial correlation decays rapidly with distance, but never truly reaches the sill. It is commonly used for phenomena where the correlation drops off quickly and the underlying process is continuous but not differentiable at the origin. {cite:p}`chiles_geostatistics_2012`

---

### Gaussian

$$
\gamma(h) = C_0 + C_1 \left( 1 - e^{-(h/a)^2} \right)
\label{eq:gaussian}
$$

The Gaussian model is characterized by a very smooth rise from the origin, with the semivariogram approaching the sill gradually. This model is appropriate for processes that are highly regular and differentiable, such as temperature or pressure fields. {cite:p}`chiles_geostatistics_2012`

---

### J-Bessel

$$
\gamma(h) = C_0 + C_1 \left( 1 - \text{J}_0(bh)\right)
\label{eq:jbessel}
$$

The J-Bessel model introduces damped oscillations into the semivariogram, which can represent periodic or cyclic spatial patterns. This is useful for modeling phenomena with repeating structures or "hole effects." {cite:p}`chiles_geostatistics_2012`<br>
*Parameter limits*: $b > 0$

### K-Bessel

$$
\gamma(h) = C_0 + C_1 \left[1 - \frac{h}{b}\text{K}_1\left(\frac{h}{b}\right)\right]
\label{eq:kbessel}
$$

Here, $b$ is a correlation length in the same units as $h$, unlike the frequency parameter used by the oscillatory Bessel models. The normalized correlation $(h/b)\text{K}_1(h/b)$ approaches one at the origin and zero at large lags. The function returns $C_0$ at zero lag. For $b=0$, its limiting value is $C$ at every positive lag. {cite:p}`chiles_geostatistics_2012`

---

(eq-legendre)=
### Legendre

$$
2 - \frac{1 - L^2}{1 - 2L \cos(\Omega) + L^2}
\label{eq:legendre}
$$

The Legendre polynomial semivariogram model is used when you need to model spatial correlation on the surface of a sphere, such as the Earth. This is especially important for global or large-scale geostatistical analyses where the curvature of the Earth cannot be ignored and the usual Euclidean distance-based models are inadequate.<br>
*Parameter limits*: $L \in (0, 1)$

Here, the parameter $\Omega$ is the angular distance between two points on a sphere (in radians) of radius $\approx$ 6378 km (Earth's radius) where:

$$
\Omega = \frac{h}{6378.137\pi}.
$$

---

### Linear

$$
\gamma(h) = C_0 + C_1 h
\label{eq:linear}
$$

The linear model describes a process where the semivariogram increases indefinitely with distance, indicating a lack of spatial stationarity. This model is typically used to represent a spatial trend or drift. {cite:p}`cressie_statistics`

---

### Linear Plateau

$$
\gamma(h) =
\begin{cases}
C_0 + C_1 \dfrac{h}{a}, & h \leq a \\
C_0 + C_1, & h > a
\end{cases}
\label{eq:linearplateau}
$$

The linear plateau model describes a spatial process where the semivariogram increases linearly with distance up to the range $a$, after which it flattens at the sill $C_0 + C_1$. This model is appropriate for phenomena where spatial correlation decreases steadily with distance until a threshold, beyond which further separation does not increase the variance. It is commonly used in soil science, hydrology, and environmental studies where a finite range and a linear rise in semivariance are observed. {cite:p}`webster_2007`

---

### Logarithmic

$$
\gamma(h) = \log(h + a)
\label{eq:log}
$$

The logarithmic semivariogram model increases slowly with distance and does not have a finite sill. It is sometimes used for spatial processes where the variance grows without bound but at a diminishing rate, such as certain economic or environmental phenomena. This model is not commonly used in geostatistics but can be useful for exploratory analysis or for fitting empirical variograms with long-range dependence. {cite:p}`chiles_geostatistics_2012`

---

### Matérn

$$
\gamma(h) = C_0 + C_1 \left[1 - \frac{2^{1-\nu}}{\Gamma(\nu)} \left(\frac{\sqrt{2\nu}h}{a}\right)^\nu \text{K}_\nu\left(\frac{\sqrt{2\nu}h}{a}\right)\right]
\label{eq:matern}
$$

The Matérn model provides a flexible approach to modeling spatial correlation. By adjusting the smoothness parameter, users can fit a wide variety of spatial behaviors, from rough to very smooth fields. {cite:p}`matern_spatial_1986`

---

### Matérn (Stein's Parameterization)

$$
\gamma(h) = 1 - \frac{2^{1-\nu}}{\Gamma(\nu)} \left( 2\sqrt{\nu} \frac{h}{a} \right)^{\nu} \text{K}_{\nu}\left( 2\sqrt{\nu} \frac{h}{a} \right)
\label{eq:stein}
$$

This is the Matérn model using Michael Stein's parameterization, which is widely used in spatial statistics. The smoothness parameter $\nu$ controls the differentiability of the process, and $a$ is the range parameter. This model generalizes the exponential and Gaussian models and is highly flexible for fitting spatial correlation. {cite:p}`stein_1999`

---

### Nugget

$$
\gamma(h) =
\begin{cases}
0 & h = 0 \\
C & h > 0
\end{cases}
\label{eq:nugget}
$$

All variation is uncorrelated, so the semivariance at positive lags equals the total sill $C$. The `nugget` argument does not add another contribution to that total. {cite:p}`cressie_statistics`

---

### Pentaspherical

$$
\gamma(h) =
\begin{cases}
C_0 + C_1 \left( \frac{15h}{8a} - \frac{5h^3}{4a^3} + \frac{3h^5}{8a^5} \right), & h \leq a \\
C_0 + C_1, & h > a
\end{cases}
\label{eq:pentaspherical}
$$

The pentaspherical model is smoother than the spherical model, with continuous derivatives up to the second order. It is useful for modeling spatial processes that exhibit a moderate degree of smoothness and a clear range beyond which correlation vanishes. {cite:p}`chiles_geostatistics_2012, webster_2007`

---

### Periodic

$$
\gamma(h) = 1 - \cos\left( \frac{2\pi h}{a} \right)
\label{eq:periodic}
$$

The periodic model produces a semivariogram with regular oscillations, suitable for phenomena with repeating spatial patterns, such as seasonal or cyclic environmental processes. {cite:p}`chiles_geostatistics_2012`

---

### Power Law

$$
\gamma(h) = C_0 + C_1 h^p
\label{eq:power}
$$

The power law model is used for fractal or scale-invariant processes, where the semivariogram increases as a power of the lag distance. This model does not have a finite sill and is appropriate for phenomena that do not exhibit a clear range. {cite:p}`goovaerts_1997`

---

### Rational Quadratic

$$
\gamma(h) = C_0 + C_1 \left( 1 - \left( 1 + \frac{h^2}{2\beta a^2} \right)^{-\beta} \right)
\label{eq:quadratic}
$$

The rational quadratic model provides polynomial decay and can interpolate between exponential and Gaussian behaviors. It is useful when the spatial process exhibits intermediate smoothness or when the empirical semivariogram does not fit standard models well. {cite:p}`goovaerts_1997`

---

### Sinc

$$
\gamma(h) = C_0 + C_1 \left( 1 - \frac{\sin(\pi h/a)}{\pi h/a} \right)
\label{eq:sinc}
$$

The sinc model generates regular oscillatory patterns, which are suitable for modeling spatial phenomena with periodic or wave-like structures. It is less commonly used but can be valuable for specialized applications. {cite:p}`chiles_geostatistics_2012`

---

### Spherical

$$
\gamma(h) =
\begin{cases}
C_0 + C_1 \left[1.5\frac{h}{a} - 0.5\left(\frac{h}{a}\right)^3\right] & 0 \leq h < a \\
C & h \geq a
\end{cases}
\label{eq:spherical}
$$

The curve reaches the total sill continuously at the range $a$. Because $C$ already includes the nugget, the plateau does not add $C_0$ again. The function uses $C_0$ as the model-curve value at zero lag, while the kriging matrix uses a zero diagonal. {cite:p}`chiles_geostatistics_2012`

---

### Spline

$$
\gamma(h) = h^2 \log(h)
\label{spline}
$$

The spline semivariogram model is used for smooth interpolation and is related to thin-plate splines in spatial statistics. It increases with distance and is especially useful for modeling spatial processes where a smooth, flexible fit is desired. This model is commonly applied in geostatistical interpolation and spatial smoothing. {cite:p}`wahba_1990`

### Stable/Exclass

$$
\gamma(h) = 1 - \exp\left( -\left( \frac{h}{a} \right)^{\alpha} \right)
\label{eq:stable}
$$

The stable/exclass (exponential class) model generalizes the exponential and Gaussian models by introducing the decay power parameter $\alpha$. When $\alpha=1$, it is the exponential model; when $\alpha=2$, it is the Gaussian model. The model is useful for spatial processes with intermediate smoothness or when empirical data do not fit standard models well. {cite:p}`montero_2015`<br>
*Parameter limits:* $\alpha \leq 2$

---

### Tetraspherical

$$
\gamma(h) =
\begin{cases}
C_0 + C_1 \left[
    \arcsin\left(\frac{h}{a}\right) + \frac{h}{a}
    \sqrt{1 - \left(
        \frac{h}{a}
        \right)^2
        } +
    \frac{2}{3} \frac{h}{a} \left(
        1 - \left(
            \frac{h}{a}
        \right)^2
    \right)^\frac{3}{2}
    \right], & 0 \leq h \leq a \\
C_0 + C_1, & h > a
\end{cases}
\label{eq:tetraspherical}
$$

The tetraspherical model is a bounded semivariogram model that provides a smooth transition from the origin to the sill, with continuous derivatives. It is suitable for spatial processes where the correlation decreases smoothly and vanishes beyond a certain range. The model is commonly used in soil science and geostatistics for fitting empirical semivariograms with moderate smoothness. {cite:p}`chiles_geostatistics_2012`

---

### Wave

$$
\gamma(h) = 1 - a \frac{\sin\left( \pi h / a \right)}{\pi h}
\label{eq:wave}
$$

The wave model generates oscillatory behavior with amplitude modulated by the parameter $a$. It is used for spatial processes with wave-like or periodic structures, such as oceanographic or atmospheric data. {cite:p}`ma_2001`

---

### Whittle's Elementary Correlation

$$
\gamma(h) = C \left\{ 1 - \frac{h}{a} \text{K}_1\left(\frac{h}{a}\right) \right\}
\label{eq:whittle}
$$

Whittle's elementary correlation model is based on the modified Bessel function of the second kind and provides a flexible approach for modeling spatial correlation. It is particularly useful for processes where the correlation decays smoothly but not as rapidly as in the exponential model. The model is often used in spatial statistics and time series analysis to represent correlations that exhibit intermediate smoothness between exponential and Gaussian behaviors. {cite:p}`whittle_1954`

---

## Composite models

### Bessel-Exponential

$$
\gamma(h) = C_0 + C_1 \left[1- \text{J}_0(b h)e^{-h/L}\right]
\label{eq:besselexponential}
$$

This model introduces periodic patterns with exponential decay, making it suitable for phenomena that display both cyclic behavior and a gradual loss of spatial correlation.

### Exponential-Cosine

$$
\gamma(h) = C_0 + C_1 \left[1 \pm \cos(b h) e^{-h/L} \right]
\label{eq:besselcosine}
$$

The minus-sign correlation product is valid in one dimension. Its isotropic use in higher dimensions requires restrictions on the frequency and decay length. The legacy plus-sign option (`enhance_semivariance=True`) is not generally a valid semivariogram, even in one dimension. Neither restriction is currently enforced by the parameter validator.

### Gaussian-Bessel

$$
\gamma(h) = C_0 + C_1 \left[1- \text{J}_0(b h)e^{-(h/L)^2}\right]
\label{eq:gaussianbessel}
$$

This model combines periodic Bessel behavior with smooth Gaussian decay, allowing for the representation of regular, cyclic spatial processes with gradual loss of correlation.

### Gaussian-Cosine

$$
\gamma(h) = C_0 + C_1 \left[1- \cos(b h)e^{-(h/L)^2}\right]
\label{eq:gaussiancosine}
$$


This correlation product is valid in one dimension. Isotropic use in higher dimensions requires checking the chosen parameters. Correct values at zero lag and at the sill do not establish covariance validity in two dimensions.

### Gaussian-Linear

$$
\gamma(h) = C_0 + C_1 \left[1- (1- b h^2) e^{-(h/L)^2}\right]
\label{eq:gaussianlinear}
$$

Despite its historical name, this model multiplies a Gaussian correlation by a quadratic term. The coefficient $b$ has inverse squared-distance units. The curve approaches the sill at large lags and can exceed it where the correlation is negative. For $b\geq0$, validity in two dimensions requires $bL^2\leq1$.

### Generalized Exponential-Linear

$$
\gamma(h) = C_0 + C_1 \left[1- (1- b h^\alpha) e^{-(h/L)^\alpha}\right]
\label{eq:exponentiallinear}
$$

The coefficient $b$ has inverse distance units raised to the power $\alpha$. This is a damped polynomial correlation, without an unbounded linear trend. For $0<\alpha\leq2$ and $b\geq0$, validity in two dimensions requires $bL^\alpha\leq\alpha/2$. The equation reduces to Gaussian-Linear when $\alpha=2$. These joint parameter restrictions are not currently enforced by the parameter validator.

### Generalized Bessel-Exponential

$$
\gamma(h) = C_0 + C_1 \left[1 - \text{J}_0(bh)e^{-(h/L)^\alpha}\right]
\label{eq:genbesselexponential}
$$

The angular frequency $b$ has inverse distance units and $L$ has distance units. For $0<\alpha\leq2$, this correlation product is valid in one or two dimensions. At $b=0$, it reduces to a powered-exponential model, including the exponential and Gaussian models at $\alpha=1$ and $\alpha=2$.

import pybamm
import numpy as np


def graphite_LGM50_ocp_Chen2020(sto):
    """
    负极开路电势曲线

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stochiometry

    Returns
    -------
    :class:`pybamm.Symbol`
        Open-circuit potential
    """

    u_eq = (
        0.9 * np.exp(-39.3631 * sto)
        - 0.0509 * np.tanh(29.8538 * (sto - 0.1234))
        - 0.04499 * np.tanh(15.2921 * (sto - 0.1))
        - 0.0177 * np.tanh(21.4708 * (sto - 0.5993))
        + 0.24898
    )

    return u_eq


def graphite_LGM50_electrolyte_exchange_current_density_Chen2020(c_e, c_s_surf, c_s_max, T):
    """
    Exchange-current density for Butler-Volmer reactions between graphite and LiPF6 in
    EC:DMC.
    负极交换电流密度

    References
    ----------
    .. [1] Chang-Hui Chen, Ferran Brosa Planella, Kieran O’Regan, Dominika Gastol, W.
    Dhammika Widanage, and Emma Kendrick. "Development of Experimental Techniques for
    Parameterization of Multi-scale Lithium-ion Battery Models." Journal of the
    Electrochemical Society 167 (2020): 080534.

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_s_surf : :class:`pybamm.Symbol`
        Particle concentration [mol.m-3]
    c_s_max : :class:`pybamm.Symbol`
        Maximum particle concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """
    m_ref_neg = pybamm.Parameter("Negative reaction rate constant")
    m_ref = 6.68e-7  # (A/m2)(m3/mol)**1.5 - includes ref concentrations  电化学反应常数
    # E_r = 35000
    # arrhenius = np.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))   #R为摩尔气体常数

    # return m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5
    return m_ref_neg * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5


def nmc_LGM50_ocp_Chen2020(sto):
    """
    正极开路电势曲线

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stochiometry

    Returns
    -------
    :class:`pybamm.Symbol`
        Open-circuit potential
    """
    u_eq = (
        4.4168*np.exp(-((sto-0.2581)/0.3964)**2)
        +0.9965*np.exp(-((sto-0.5606)/0.1597)**2)
        +0.2889*np.exp(-((sto-0.6610)/0.1128)**2)
        +0.0530*np.exp(-((sto-0.7158)/0.2024)**2)
        +2.6335*np.exp(-((sto-0.8116)/0.1724)**2)
        +0.3453*np.exp(-((sto- 0.8688)/0.0790)**2)
        +0.03*np.exp(-((sto-0.8946)/0.0410)**2)
        +0.01*np.exp(-((sto-0.8789)/ 0.1103)**2)
        +1.9785*np.exp(-((sto-0.9818)/0.0915)**2)
    )    

    return u_eq


def nmc_LGM50_electrolyte_exchange_current_density_Chen2020(c_e, c_s_surf, c_s_max, T):
    """
    Exchange-current density for Butler-Volmer reactions between NMC and LiPF6 in
    EC:DMC.
    正极交换电流密度

    References
    ----------
    .. [1] Chang-Hui Chen, Ferran Brosa Planella, Kieran O’Regan, Dominika Gastol, W.
    Dhammika Widanage, and Emma Kendrick. "Development of Experimental Techniques for
    Parameterization of Multi-scale Lithium-ion Battery Models." Journal of the
    Electrochemical Society 167 (2020): 080534.

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_s_surf : :class:`pybamm.Symbol`
        Particle concentration [mol.m-3]
    c_s_max : :class:`pybamm.Symbol`
        Maximum particle concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """
    m_ref_pos = pybamm.Parameter("Positive reaction rate constant")
    m_ref = 5e-6  # (A/m2)(m3/mol)**1.5 - includes ref concentrations
    # E_r = 17800
    # arrhenius = np.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    # return m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5
    return m_ref_pos * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5


def electrolyte_diffusivity_Nyman2008(c_e, T):
    """
    Diffusivity of LiPF6 in EC:EMC (3:7) as a function of ion concentration. The data
    comes from [1]
    电解液扩散系数

    References
    ----------
    .. [1] A. Nyman, M. Behm, and G. Lindbergh, "Electrochemical characterisation and
    modelling of the mass transport phenomena in LiPF6-EC-EMC electrolyte,"
    Electrochim. Acta, vol. 53, no. 22, pp. 6356–6365, 2008.

    Parameters
    ----------
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """

    D_c_e = 8.794e-11 * (c_e / 1000) ** 2 - 3.972e-10 * (c_e / 1000) + 4.862e-10

    # Nyman et al. (2008) does not provide temperature dependence

    return D_c_e


def electrolyte_conductivity_Nyman2008(c_e, T):
    """
    Conductivity of LiPF6 in EC:EMC (3:7) as a function of ion concentration. The data
    comes from [1].
    电解液电导率

    References
    ----------
    .. [1] A. Nyman, M. Behm, and G. Lindbergh, "Electrochemical characterisation and
    modelling of the mass transport phenomena in LiPF6-EC-EMC electrolyte,"
    Electrochim. Acta, vol. 53, no. 22, pp. 6356–6365, 2008.

    Parameters
    ----------
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """

    sigma_e = (
        0.1297 * (c_e / 1000) ** 3 - 2.51 * (c_e / 1000) ** 1.5 + 3.329 * (c_e / 1000)
    )

    # Nyman et al. (2008) does not provide temperature dependence

    return sigma_e


# Call dict via a function to avoid errors when editing in place
def get_parameter_values():
    """
    Parameters for an LG M50 cell, from the paper :footcite:t:`Chen2020` and references
    therein.

    SEI parameters are example parameters for SEI growth from the papers
    :footcite:t:`Ramadass2004`, :footcite:t:`ploehn2004solvent`,
    :footcite:t:`single2018identifying`, :footcite:t:`safari2008multimodal`, and
    :footcite:t:`Yang2017`

    .. note::
        This parameter set does not claim to be representative of the true parameter
        values. Instead these are parameter values that were used to fit SEI models to
        observed experimental data in the referenced papers.
    """

    return {
        "chemistry": "lithium_ion",
        # # sei
        # "Ratio of lithium moles to SEI moles": 2.0,
        # "Inner SEI reaction proportion": 0.5,
        # "Inner SEI partial molar volume [m3.mol-1]": 9.585e-05,
        # "Outer SEI partial molar volume [m3.mol-1]": 9.585e-05,
        # "SEI reaction exchange current density [A.m-2]": 1.5e-07,
        # "SEI resistivity [Ohm.m]": 200000.0,
        # "Outer SEI solvent diffusivity [m2.s-1]": 2.5000000000000002e-22,
        # "Bulk solvent concentration [mol.m-3]": 2636.0,
        # "Inner SEI open-circuit potential [V]": 0.1,
        # "Outer SEI open-circuit potential [V]": 0.8,
        # "Inner SEI electron conductivity [S.m-1]": 8.95e-14,
        # "Inner SEI lithium interstitial diffusivity [m2.s-1]": 1e-20,
        # "Lithium interstitial reference concentration [mol.m-3]": 15.0,
        # "Initial inner SEI thickness [m]": 2.5e-09,
        # "Initial outer SEI thickness [m]": 2.5e-09,
        # "EC initial concentration in electrolyte [mol.m-3]": 4541.0,
        # "EC diffusivity [m2.s-1]": 2e-18,
        # "SEI kinetic rate constant [m.s-1]": 1e-12,
        # "SEI open-circuit potential [V]": 0.4,
        # "SEI growth activation energy [J.mol-1]": 0.0,
        # "Negative electrode reaction-driven LAM factor [m3.mol-1]": 0.0,
        # "Positive electrode reaction-driven LAM factor [m3.mol-1]": 0.0,
        
        # cell
        # "Negative current collector thickness [m]": 1.2e-05,
        "Negative electrode thickness [m]": 6.7e-05,
        "Separator thickness [m]": 1.6e-05,
        "Positive electrode thickness [m]": 6.6e-05,
        # "Positive current collector thickness [m]": 1.6e-05,
        "Electrode height [m]": 0.17,
        "Electrode width [m]": 0.08,
        # "Cell cooling surface area [m2]": 0.00531,
        # "Cell volume [m3]": 1.338e-04,
        # "Cell thermal expansion coefficient [m.K-1]": 1.1e-06,
        "Negative current collector conductivity [S.m-1]": 58411000.0,
        "Positive current collector conductivity [S.m-1]": 36914000.0,
        "Negative current collector density [kg.m-3]": 8960.0,
        "Positive current collector density [kg.m-3]": 2700.0,
        "Negative current collector specific heat capacity [J.kg-1.K-1]": 385.0,
        "Positive current collector specific heat capacity [J.kg-1.K-1]": 897.0,
        "Negative current collector thermal conductivity [W.m-1.K-1]": 401.0,
        "Positive current collector thermal conductivity [W.m-1.K-1]": 237.0,
        "Nominal cell capacity [A.h]": 5.2,
        "Current function [A]": 5.0,
        "Contact resistance [Ohm]": 0,
       
        # negative electrode
        "Negative electrode conductivity [S.m-1]": 200,
        "Maximum concentration in negative electrode [mol.m-3]": 60219.86,
        "Negative electrode diffusivity [m2.s-1]": 2.2332e-10,
        "Negative electrode OCP [V]": graphite_LGM50_ocp_Chen2020,
        "Negative electrode porosity": 0.0323,
        "Negative electrode active material volume fraction": 0.4357,
        "Negative particle radius [m]": 4.8626e-06,
        "Negative electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Negative electrode Bruggeman coefficient (electrode)": 0,
        "Negative electrode charge transfer coefficient": 0.5,
        "Negative electrode double-layer capacity [F.m-2]": 0.2,
        "Negative electrode exchange-current density [A.m-2]"
        "": graphite_LGM50_electrolyte_exchange_current_density_Chen2020,
        "Negative electrode density [kg.m-3]": 1657.0,
        "Negative electrode specific heat capacity [J.kg-1.K-1]": 700.0,
        "Negative electrode thermal conductivity [W.m-1.K-1]": 1.7,
        "Negative electrode OCP entropic change [V.K-1]": 0.0,
        "Negative reaction rate constant": 4.6107e-4,
        
       
        # positive electrode
        "Positive electrode conductivity [S.m-1]": 1.2,
        "Maximum concentration in positive electrode [mol.m-3]": 19768.43,
        "Positive electrode diffusivity [m2.s-1]": 7.3706e-10,
        "Positive electrode OCP [V]": nmc_LGM50_ocp_Chen2020,
        "Positive electrode porosity": 0.4973,
        "Positive electrode active material volume fraction": 0.6664,
        "Positive particle radius [m]": 1.2822e-07,
        "Positive electrode Bruggeman coefficient (electrolyte)": 1.5,
        "Positive electrode Bruggeman coefficient (electrode)": 0,
        "Positive electrode charge transfer coefficient": 0.5,
        "Positive electrode double-layer capacity [F.m-2]": 0.2,
        "Positive electrode exchange-current density [A.m-2]"
        "": nmc_LGM50_electrolyte_exchange_current_density_Chen2020,
        "Positive electrode density [kg.m-3]": 3262.0,
        "Positive electrode specific heat capacity [J.kg-1.K-1]": 700.0,
        "Positive electrode thermal conductivity [W.m-1.K-1]": 2.1,
        "Positive electrode OCP entropic change [V.K-1]": 0.0,
        "Positive reaction rate constant":2.91e-5,
        
        # separator
        "Separator porosity": 0.2851,
        "Separator Bruggeman coefficient (electrolyte)": 1.5,
        "Separator density [kg.m-3]": 397.0,
        "Separator specific heat capacity [J.kg-1.K-1]": 700.0,
        "Separator thermal conductivity [W.m-1.K-1]": 0.16,
        
        # electrolyte
        "Initial concentration in electrolyte [mol.m-3]": 940.08,
        "Cation transference number": 0.2594,
        "Thermodynamic factor": 1.0,
        # "Electrolyte diffusivity [m2.s-1]": electrolyte_diffusivity_Nyman2008,
        "Electrolyte diffusivity [m2.s-1]": 3.1717e-10,
        "Electrolyte conductivity [S.m-1]": 5.7478,
        # "Electrolyte conductivity [S.m-1]": electrolyte_conductivity_Nyman2008,
        
        # experiment
        "Reference temperature [K]": 298.15,
        "Total heat transfer coefficient [W.m-2.K-1]": 10.0,
        "Ambient temperature [K]": 298.15,
        "Number of electrodes connected in parallel to make a cell": 26,
        "Number of cells connected in series to make a battery": 1.0,
        "Lower voltage cut-off [V]": 2.7,
        "Upper voltage cut-off [V]": 4.2,
        "Open-circuit voltage at 0% SOC [V]": 2.7,
        "Open-circuit voltage at 100% SOC [V]": 4.2,
        "Initial concentration in negative electrode [mol.m-3]": 57473.83,
        "Initial concentration in positive electrode [mol.m-3]": 6897.21,
        "Initial temperature [K]": 298.15,

    }

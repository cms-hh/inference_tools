# coding: utf-8
# model for new channels development && resonant

from collections import OrderedDict, defaultdict

from HiggsAnalysis.CombinedLimit.PhysicsModel import PhysicsModelBase

__all__ = [
    # model
    "HHModelBase", "HHModel", "create_model", "model_dummy",
    # xsec helpers
    "create_hh_xsec_func",
]

no_value = object()


class HHModelBase(PhysicsModelBase):
    """
    Base class for HH physics models providing a common interface for subclasses such as the default
    HH model or a potential EFT model (e.g. kt-kl-C2).
    """

    # pois with initial (SM) value, start and stop
    # to be defined by subclasses
    R_POIS = OrderedDict()
    K_POIS = OrderedDict()

    def __init__(self, name):
        super(HHModelBase, self).__init__()

        # attributes
        self.name = name

        # names and values of physics options
        self.hh_options = OrderedDict()

        # actual r and k pois, depending on used formulae and profiling options, set in reset_pois
        self.r_pois = None
        self.k_pois = None

        # mapping of (formula, sample) -> expression name that models the linear sample scales
        self.r_expressions = {}

        # nested mapping of formula -> sample -> matched processes for book keeping
        self.process_scales = defaultdict(lambda: defaultdict(set))

    def register_opt(self, name, default, is_flag=False):
        """
        Registers a physics option which is automatically parsed and set by
        :py:meth:`setPhysicsOptions`. Example:

        .. code-block:: python

            register_opt("myName", "some_default_value")
            # -> parses "--physics-option myName='other_value'"
        """
        self.hh_options[name] = {
            "value": default,
            "is_flag": is_flag,
        }

    def set_opt(self, name, value):
        """
        Sets the value of a physics option named *name*, previoulsy registered with
        :py:meth:`register_opt`, to *value*.
        """
        self.hh_options[name]["value"] = value

    def opt(self, name, default=no_value):
        """
        Helper to get the value of a physics option defined by *name* with an optional *default*
        value that is returned when no option with that *name* is registered.
        """
        if name in self.hh_options or default == no_value:
            return self.hh_options[name]["value"]

        return default

    def setPhysicsOptions(self, options):
        """
        Hook called by the super class to parse physics options received externally, e.g. via
        ``--physics-option`` or ``--PO``.
        """
        # split by "=" and check one by one
        pairs = [opt.split("=", 1) for opt in options if "=" in opt]
        for name, value in pairs:
            if name not in self.hh_options:
                print("[WARNING] unknown physics option '{}'".format(name))
                continue

            if self.hh_options[name]["is_flag"]:
                # boolean flag
                value = value.lower() in ["yes", "true", "1"]
            else:
                # string value, catch special cases
                value = None if value.lower() in ["", "none"] else value

            self.set_opt(name, value)
            print("[INFO] using model option {} = {}".format(name, value))

        # since settings might have changed, reset pois again
        self.reset_pois()

    def reset_pois(self):
        """
        Sets the instance-level :py:attr:`r_pois` based on registered
        formulae.
        """
        # r pois
        self.r_pois = OrderedDict()
        for p, v in self.R_POIS.items():
            keep = p == "r"
            if keep:
                self.r_pois[p] = v

        # k pois
        self.k_pois = OrderedDict()
        for p, v in self.K_POIS.items():
            if keep:
                self.k_pois[p] = v

    def get_formulae(self, xs_only=False):
        """
        Method that returns a dictionary of all used :py:class:`HHFormula` instances, mapped to
        their attribute names. When *xs_only* is *True*, only those formulae are returned that
        should enter cross section calculations.
        To be implemented in subclasses.
        """
        raise NotImplementedError

    @property
    def model_builder(self):
        # for compatibility
        return self.modelBuilder

    def make_expr(self, *args, **kwargs):
        """
        Shorthand for :py:meth:`model_builder.factory_`.
        """
        return self.model_builder.factory_(*args, **kwargs)

    def get_expr(self, *args, **kwargs):
        """
        Shorthand for :py:meth:`model_builder.out.function`.
        """
        return self.model_builder.out.function(*args, **kwargs)

    def make_var(self, *args, **kwargs):
        """
        Shorthand for :py:meth:`model_builder.doVar`.
        """
        return self.model_builder.doVar(*args, **kwargs)

    def get_var(self, *args, **kwargs):
        """
        Shorthand for :py:meth:`model_builder.out.var`.
        """
        return self.model_builder.out.var(*args, **kwargs)

    def make_set(self, *args, **kwargs):
        """
        Shorthand for :py:meth:`model_builder.doSet`.
        """
        return self.model_builder.doSet(*args, **kwargs)

    def done(self):
        """
        Hook called by the super class after the workspace is created.

        Here, we make sure that each formula will function as expected by checking that each of its
        samples is matched by exactly one process. When no processes was matched for _any_ sample of
        a formula, the scaling is disabled and the underlying r POI will have no effect.
        """
        super(HHModelBase, self).done()

        errors = []

        if errors:
            raise Exception("\n".join(errors))


class HHModel(HHModelBase):
    """
    Models empty to allow start develop new HH-related channels.
    Also serves to resonant (eg pulls computation)
    """

    # pois with initial (SM) value, start and stop
    R_POIS = OrderedDict([
        ("r", (1, -20, 20)),
    ])

    def __init__(self, name,):
        super(HHModel, self).__init__(name)

        # register options
        self.register_opt("doNNLOscaling", True, is_flag=True)
        self.register_opt("doklDependentUnc", True, is_flag=True)
        self.register_opt("doBRscaling", True, is_flag=True)
        self.register_opt("doHscaling", True, is_flag=True)

        # reset instance-level pois
        self.reset_pois()

    def reset_pois(self):
        super(HHModel, self).reset_pois()

        # remove profiled r pois
        for p in list(self.r_pois):
            if self.opt("doProfile" + p.replace("_", ""), False):
                del self.r_pois[p]

    def get_formulae(self, xs_only=False):
        formulae = OrderedDict()
        return formulae

    def _create_hh_xsec_func(self, *args, **kwargs):
        # forward to the modul-level implementation
        return create_hh_xsec_func(*args, **kwargs)

    def create_hh_xsec_func(self, **kwargs):
        """
        Returns a function that can be used to compute cross sections, based on all formulae
        returned by :py:meth:`get_formulae` with *xs_only* set to *True*.
        """
        _kwargs = self.get_formulae(xs_only=True)
        _kwargs.update(kwargs)
        return self._create_hh_xsec_func(**_kwargs)

    def doParametersOfInterest(self):
        """
        Hook called by the super class to add parameters (of interest) to the model.

        Here, we add all r and k POIs depending on profiling options, define the POI group and
        re-initialize the MH parameter. By default, the main r POI will be the only floating one,
        whereas the others are either fixed or fully profiled.
        """
        # first, add all known r and k POIs
        for p in self.R_POIS:
            value, start, stop = self.r_pois.get(p, self.R_POIS[p])
            self.make_var("{}[{},{},{}]".format(p, value, start, stop))

        # make certain r parameters pois, freeze all but the main r
        pois = []
        for p, (value, start, stop) in self.r_pois.items():
            if p != "r":
                self.get_var(p).setConstant(True)
            pois.append(p)

        # define the POI group
        self.make_set("POI", ",".join(pois))
        print("using POIs {}".format(",".join(pois)))

        # set or redefine the MH variable on which some of the BRs depend
        if not self.options.mass:
            raise Exception(
                "invalid mass value '{}', please provide a valid value using the "
                "--mass option".format(self.options.mass),
            )
        if self.get_var("MH"):
            self.get_var("MH").removeRange()
            self.get_var("MH").setVal(self.options.mass)
        else:
            self.make_var("MH[{}]".format(self.options.mass))
        self.get_var("MH").setConstant(True)

    def preProcessNuisances(self, nuisances):
        """
        Hook called by the super class before nuisances are processed.

        Here, we make sure that all custom parameters are properly added to the model:

        - kl-dependent ggf theory uncertainty
        - parameters of coupling modifiers when profiling
        """
        # enable profiling of r and k POIs with a configurable prior when requested
        for p in list(self.R_POIS.keys()):
            value = self.opt("doProfile" + p.replace("_", ""), False)
            if not value:
                continue

            # get the prior and add it
            prior, width = value.split(",", 1) if "," in value else (value, None)
            if prior == "flat":
                self.model_builder.DC.flatParamNuisances[p] = True
                print("adding flat prior for parameter {}".format(p))
            elif prior == "gauss":
                nuisances.append((p, False, "param", ["1", width, "[-7,7]"], []))
                print("adding gaussian prior for parameter {} with width {}".format(p, width))
            else:
                raise Exception("unknown prior '{}' for parameter {}".format(prior, p))


def create_model(name, ggf=None, vbf=None, vhh=None, **kwargs):
    """
    Returns a new :py:class:`HHModel` All
    additional *kwargs* are forwarded to the model constructor.
    """
    # create the return the model
    return HHModel(
        name=name,
        **kwargs  # noqa
    )


model_dummy= create_model("model_dummy",)

####################################################################################################
# dummy cross section helpers
####################################################################################################


def create_hh_xsec_func(ggf_formula=None, vbf_formula=None, vhh_formula=None):
    return 0

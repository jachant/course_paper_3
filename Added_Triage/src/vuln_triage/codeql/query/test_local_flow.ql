import python

from Expr e
where
  e.getLocation().getFile().getRelativePath().matches("%physcialsim_metrics.py")
select e, e.getLocation()
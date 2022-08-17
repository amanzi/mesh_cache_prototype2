#pragma once


bool inline
close(double gold, double test, double rtol, double atol=0)
{
  if (std::abs(gold - test)/(atol + rtol * gold) < 1) return true;
  return false;
}
